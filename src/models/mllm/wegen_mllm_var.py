import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.models.tokenizer.eva_vision import AvgPoolProjector, LinearProjector
from src.models.mllm.wegen_mllm_joint_llm_sd import ContinuousLVLMJointLLMSD
from transformers import LogitsProcessorList
from .generation import AutoImageTokenGenerationProcessor
from .utils import load_zero3_checkpoint
from .wegen_mllm import ContinuousLVLM


BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


def get_cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


class ContinuousLVLMJointLLMSDVAR(ContinuousLVLMJointLLMSD):
    def __init__(self, tied_projector=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tied_projector = tied_projector
        if tied_projector:
            assert isinstance(self.input_resampler, AvgPoolProjector)
            assert isinstance(self.output_resampler, LinearProjector)
            self.output_resampler.proj.weight = nn.Parameter(self.input_resampler.proj.weight.T.contiguous())

    def forward(self, input_ids, attention_mask, labels, image_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
                ids_cmp_mask, patch_positions=None, images_aug=None):
        
        assert images_aug is not None

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            image_embeds_cmp = image_embeds[embeds_cmp_mask]  # num_imgs_in_batch x nq_in x dim_in, 4 x 64 x 4096
            if patch_positions is not None:
                patch_positions = patch_positions[embeds_cmp_mask]
        

        if image_embeds is not None and image_embeds_cmp.shape[0] > 0:
            image_embeds_lm = self.input_resampler(image_embeds_cmp)  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            # TODO[huangzp]: check it 
            # if self.add_patch_pos and patch_positions is not None:
            #     # assert patch_positions is not None
            #     patch_positions = patch_positions.to(
            #                                 image_embeds_lm
            #                                 ) 
            #     rel_pos_embed = torch.mm(torch.cat([patch_positions, 1-patch_positions], dim=-1)/2, self.patch_pos_embed).unsqueeze(1)
            #     image_embeds_lm = image_embeds_lm + rel_pos_embed
            has_image_cmp = True

        has_image_input = image_embeds is not None and embeds_cmp_mask.sum().item() > 0
        has_image_output = image_embeds is not None and embeds_gen_mask.sum().item() > 0

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0
        if has_image_output:
            rec_loss = 0.0
            target_embeds = image_embeds[embeds_gen_mask]
            target_embeds_lm = self.input_resampler(target_embeds)
            input_embeds[ids_gen_mask] = target_embeds_lm.reshape(-1, dim)
            
            
        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']
        # print(f'lm_loss: {lm_loss}')

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096
        
        rec_loss_attr_value = {}
        if has_image_output:
            if self.vit_down:
                b, n, c = target_embeds.shape
                sqrt_n = int(n**0.5)
                target_embeds = target_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
                stride = int(sqrt_n // (self.n_query ** 0.5))
                target_embeds = F.avg_pool2d(target_embeds, kernel_size=(stride, stride), stride=stride)
                target_embeds = target_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
            target_image_aug = images_aug[embeds_gen_mask]
            num_imgs_for_rec = target_embeds.shape[0]
            # shift ids_gen_mask to left by 1 to get the output ids mask
            output_gens_mask = ids_gen_mask.clone()
            output_gens_mask[:, :-1] = ids_gen_mask[:, 1:]
            output_gens_mask[:, -1] = False
            output_image_embeds = last_hidden_state[output_gens_mask].view(num_imgs_for_rec, -1, dim)  # 128 x 4096 -> 2 x 64 x 4096
            recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 256 x 4096
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                if rec_loss_attr_cur == 'mse_loss':
                    # rec_loss = self.mse_loss(recon_image_embeds, target_embeds.detach())
                    mse_loss = F.mse_loss(recon_image_embeds, target_embeds.detach()) # for zero3 compatibility
                    rec_loss += rec_loss_attr_weight_cur * mse_loss
                    rec_loss_attr_value['mse_loss'] = mse_loss
                elif rec_loss_attr_cur == 'cosine_loss':
                    cosine_loss = get_cosine_loss(recon_image_embeds, target_embeds.detach())
                    rec_loss += rec_loss_attr_weight_cur * cosine_loss
                    rec_loss_attr_value['cosine_loss'] = cosine_loss
                elif rec_loss_attr_cur == 'diffusion_loss':
                    diffusion_loss = self.adapter.train_pipeline(recon_image_embeds, target_image_aug, recon_image_embeds.device, recon_image_embeds.dtype)['total_loss']
                    rec_loss += rec_loss_attr_weight_cur * diffusion_loss
                    rec_loss_attr_value['diffusion_loss'] = diffusion_loss
                elif rec_loss_attr_cur == 'inside_mse_loss':
                    with torch.no_grad():
                        inside_target_embeds = self.input_resampler(target_embeds.detach())
                    inside_mse_loss = F.mse_loss(output_image_embeds, inside_target_embeds)
                    rec_loss += rec_loss_attr_weight_cur * inside_mse_loss
                    rec_loss_attr_value['inside_mse_loss'] = inside_mse_loss
                else:
                    raise NotImplementedError
        else:
            output_image_embeds = torch.randn(1, self.input_resampler.num_queries,
                                              self.input_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype) + 0.0 * last_hidden_state[0, :self.input_resampler.num_queries, :]
            recon_image_embeds = self.output_resampler(output_image_embeds)

            rec_loss = 0.0 * recon_image_embeds.sum()
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                loss_value = 0.0 * recon_image_embeds.sum()
                rec_loss_attr_value[rec_loss_attr_cur] = loss_value
                if rec_loss_attr_cur in ['cosine_loss', 'diffusion_loss', 'inside_mse_loss']:
                    rec_loss += rec_loss_attr_weight_cur * loss_value
                elif rec_loss_attr_cur != 'mse_loss':
                    raise NotImplementedError

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss
        
        ret = {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'rec_loss': rec_loss,
        }
        ret.update(rec_loss_attr_value)
        # if has_image_output:
        #     ret['recon_image_embeds'] = recon_image_embeds
        #     ret['target_embeds'] = target_embeds

        return ret
    
    def generate(self,
                 tokenizer,
                 prompt=None,
                 input_ids=None,
                 image_embeds=None,
                 embeds_cmp_mask=None,
                 ids_cmp_mask=None,
                 logits_processor=None,
                 num_img_gen_tokens=64,
                 temperature=0.7,
                 num_beams=1,
                 max_new_tokens=120,
                 top_p=0.5,
                 dtype=torch.float16,
                 device='cuda',
                 patch_positions=None,
                 var_cfg_guidance_scale=1.0,
                 ):
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
            logits_processor.append(
                AutoImageTokenGenerationProcessor(tokenizer=tokenizer, num_img_gen_tokens=num_img_gen_tokens))

        if prompt is not None:
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids)

        input_ids = input_ids.to(device=device)
        input_embeds = self.llm.get_input_embeddings()(input_ids)
        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            assert embeds_cmp_mask is not None and ids_cmp_mask is not None
            with torch.no_grad():
                image_embeds_lm = self.input_resampler(image_embeds)
                if self.add_patch_pos:
                    assert patch_positions is not None
                    patch_positions = patch_positions.to(
                                                image_embeds_lm
                                                ) 
                    rel_pos_embed = torch.mm(torch.cat([patch_positions, 1-patch_positions], dim=-1)/2, self.patch_pos_embed).unsqueeze(1)
                    image_embeds_lm = image_embeds_lm + rel_pos_embed
            #print(input_embeds.shape, ids_cmp_mask.shape, image_embeds_lm.shape, embeds_cmp_mask.shape)
            input_embeds[ids_cmp_mask] = image_embeds_lm[embeds_cmp_mask].view(-1, dim)

        # TODO[huangzp]: add the input and output projector to the generation config
        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False,
            # 'use_cache': False,
            # 'past_key_values': None,
            'input_projector': self.input_resampler.proj,
            'output_projector': self.output_resampler.proj,
            'var_cfg_guidance_scale': var_cfg_guidance_scale,
        }

        # generate_ids = self.llm.generate(input_ids=input_ids, **generation_config)
        output = self.llm.generate(input_ids=input_ids,
                                   inputs_embeds=input_embeds,
                                   output_hidden_states=True,
                                   return_dict_in_generate=True,
                                   logits_processor=logits_processor,
                                   **generation_config)

        generate_ids = output.sequences[0][input_ids.shape[1]:]
        generate_id_list = generate_ids.tolist()
        boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
        eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

        last_hidden_states = torch.cat([hidden_state[-1] for hidden_state in output.hidden_states],
                                       dim=1)[0, input_ids.shape[1]:, :]
        # if use_cache is False:
        # last_hidden_states = output.hidden_states[-1][-1][0, input_ids.shape[1]:, :]

        eoi_indices = torch.where(generate_ids == eoi_token_id)[0].tolist()
        num_gen_imgs = len(eoi_indices)
        text_mask = torch.ones_like(generate_ids, dtype=torch.bool)
        has_img_output = num_gen_imgs > 0
        if has_img_output:
            img_gen_feats = []
            for eoi_idx in eoi_indices:
                img_gen_feats.append(last_hidden_states[eoi_idx - num_img_gen_tokens:eoi_idx])
                text_mask[eoi_idx - num_img_gen_tokens:eoi_idx] = False

            img_gen_feats = torch.stack(img_gen_feats)
            img_gen_feat = self.output_resampler(img_gen_feats)
        else:
            img_gen_feats = None
            img_gen_feat = None

        text_mask[generate_ids == boi_token_id] = False
        generate_ids = generate_ids[text_mask]
        generate_text = tokenizer.decode(generate_ids, skip_special_tokens=False)
        
        
        # full text
        full_ids = output.sequences[0]
        full_text = tokenizer.decode(full_ids, skip_special_tokens=False)
        

        return {
            'text': generate_text,
            'has_img_output': has_img_output,
            'img_gen_feat': img_gen_feat,
            'num_gen_imgs': num_gen_imgs,
            'full_text': full_text,
            'output_lm_gen_feat': img_gen_feats,
        }
        

class ContinuousLVLMJointLLMSDVARNoise(ContinuousLVLMJointLLMSDVAR):
    def __init__(self, noise_scale, *args, **kwargs):
        self.noise_scale = noise_scale
        super().__init__(*args, **kwargs)

    def forward(self, input_ids, attention_mask, labels, image_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
                ids_cmp_mask, patch_positions=None, images_aug=None):
        
        assert images_aug is not None

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            image_embeds_cmp = image_embeds[embeds_cmp_mask]  # num_imgs_in_batch x nq_in x dim_in, 4 x 64 x 4096
            if patch_positions is not None:
                patch_positions = patch_positions[embeds_cmp_mask]
        

        if image_embeds is not None and image_embeds_cmp.shape[0] > 0:
            image_embeds_lm = self.input_resampler(image_embeds_cmp)  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            # TODO[huangzp]: check it 
            # if self.add_patch_pos and patch_positions is not None:
            #     # assert patch_positions is not None
            #     patch_positions = patch_positions.to(
            #                                 image_embeds_lm
            #                                 ) 
            #     rel_pos_embed = torch.mm(torch.cat([patch_positions, 1-patch_positions], dim=-1)/2, self.patch_pos_embed).unsqueeze(1)
            #     image_embeds_lm = image_embeds_lm + rel_pos_embed
            has_image_cmp = True

        has_image_input = image_embeds is not None and embeds_cmp_mask.sum().item() > 0
        has_image_output = image_embeds is not None and embeds_gen_mask.sum().item() > 0

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0
        if has_image_output:
            rec_loss = 0.0
            target_embeds = image_embeds[embeds_gen_mask]
            target_embeds_lm = self.input_resampler(target_embeds)
            target_embeds_lm = target_embeds_lm.reshape(-1, dim)
            input_embeds[ids_gen_mask] = target_embeds_lm
            # add noise
            noise = torch.randn_like(target_embeds_lm, dtype=target_embeds_lm.dtype, device=target_embeds_lm.device) * self.noise_scale
            input_embeds[ids_gen_mask] += noise
            
            
            
        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']
        # print(f'lm_loss: {lm_loss}')

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096
        
        rec_loss_attr_value = {}
        if has_image_output:
            if self.vit_down:
                b, n, c = target_embeds.shape
                sqrt_n = int(n**0.5)
                target_embeds = target_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
                stride = int(sqrt_n // (self.n_query ** 0.5))
                target_embeds = F.avg_pool2d(target_embeds, kernel_size=(stride, stride), stride=stride)
                target_embeds = target_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
            target_image_aug = images_aug[embeds_gen_mask]
            num_imgs_for_rec = target_embeds.shape[0]
            # shift ids_gen_mask to left by 1 to get the output ids mask
            output_gens_mask = ids_gen_mask.clone()
            output_gens_mask[:, :-1] = ids_gen_mask[:, 1:]
            output_gens_mask[:, -1] = False
            output_image_embeds = last_hidden_state[output_gens_mask].view(num_imgs_for_rec, -1, dim)  # 128 x 4096 -> 2 x 64 x 4096
            recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 256 x 4096
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                if rec_loss_attr_cur == 'mse_loss':
                    # rec_loss = self.mse_loss(recon_image_embeds, target_embeds.detach())
                    mse_loss = F.mse_loss(recon_image_embeds, target_embeds.detach()) # for zero3 compatibility
                    rec_loss += rec_loss_attr_weight_cur * mse_loss
                    rec_loss_attr_value['mse_loss'] = mse_loss
                elif rec_loss_attr_cur == 'cosine_loss':
                    cosine_loss = get_cosine_loss(recon_image_embeds, target_embeds.detach())
                    rec_loss += rec_loss_attr_weight_cur * cosine_loss
                    rec_loss_attr_value['cosine_loss'] = cosine_loss
                elif rec_loss_attr_cur == 'diffusion_loss':
                    diffusion_loss = self.adapter.train_pipeline(recon_image_embeds, target_image_aug, recon_image_embeds.device, recon_image_embeds.dtype)['total_loss']
                    rec_loss += rec_loss_attr_weight_cur * diffusion_loss
                    rec_loss_attr_value['diffusion_loss'] = diffusion_loss
                else:
                    raise NotImplementedError
        else:
            output_image_embeds = torch.randn(1, self.input_resampler.num_queries,
                                              self.input_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype) + 0.0 * last_hidden_state[0, :self.input_resampler.num_queries, :]
            recon_image_embeds = self.output_resampler(output_image_embeds)

            rec_loss = 0.0 * recon_image_embeds.sum()
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                loss_value = 0.0 * recon_image_embeds.sum()
                rec_loss_attr_value[rec_loss_attr_cur] = loss_value
                if rec_loss_attr_cur in ['cosine_loss', 'diffusion_loss', 'inside_mse_loss']:
                    rec_loss += rec_loss_attr_weight_cur * loss_value
                elif rec_loss_attr_cur != 'mse_loss':
                    raise NotImplementedError

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss
        
        ret = {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'rec_loss': rec_loss,
        }
        ret.update(rec_loss_attr_value)
        # if has_image_output:
        #     ret['recon_image_embeds'] = recon_image_embeds
        #     ret['target_embeds'] = target_embeds

        return ret
    
    
class ContinuousLVLMJointLLMSDVARAddLearnQuery(ContinuousLVLMJointLLMSDVAR):

    def forward(self, input_ids, attention_mask, labels, image_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
                ids_cmp_mask, patch_positions=None, images_aug=None):
        
        assert images_aug is not None

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            image_embeds_cmp = image_embeds[embeds_cmp_mask]  # num_imgs_in_batch x nq_in x dim_in, 4 x 64 x 4096
            if patch_positions is not None:
                patch_positions = patch_positions[embeds_cmp_mask]
        

        if image_embeds is not None and image_embeds_cmp.shape[0] > 0:
            image_embeds_lm = self.input_resampler(image_embeds_cmp)  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            # TODO[huangzp]: check it 
            # if self.add_patch_pos and patch_positions is not None:
            #     # assert patch_positions is not None
            #     patch_positions = patch_positions.to(
            #                                 image_embeds_lm
            #                                 ) 
            #     rel_pos_embed = torch.mm(torch.cat([patch_positions, 1-patch_positions], dim=-1)/2, self.patch_pos_embed).unsqueeze(1)
            #     image_embeds_lm = image_embeds_lm + rel_pos_embed
            has_image_cmp = True

        has_image_input = image_embeds is not None and embeds_cmp_mask.sum().item() > 0
        has_image_output = image_embeds is not None and embeds_gen_mask.sum().item() > 0

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0
        if has_image_output:
            rec_loss = 0.0
            target_embeds = image_embeds[embeds_gen_mask]
            target_embeds_lm = self.input_resampler(target_embeds)
            target_embeds_lm = target_embeds_lm.reshape(-1, dim)
            # learnable query + target_embeds_lm
            input_embeds[ids_gen_mask] += target_embeds_lm
            
        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']
        # print(f'lm_loss: {lm_loss}')

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096
        
        rec_loss_attr_value = {}
        if has_image_output:
            if self.vit_down:
                b, n, c = target_embeds.shape
                sqrt_n = int(n**0.5)
                target_embeds = target_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
                stride = int(sqrt_n // (self.n_query ** 0.5))
                target_embeds = F.avg_pool2d(target_embeds, kernel_size=(stride, stride), stride=stride)
                target_embeds = target_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()
            target_image_aug = images_aug[embeds_gen_mask]
            num_imgs_for_rec = target_embeds.shape[0]
            # shift ids_gen_mask to left by 1 to get the output ids mask
            output_gens_mask = ids_gen_mask.clone()
            output_gens_mask[:, :-1] = ids_gen_mask[:, 1:]
            output_gens_mask[:, -1] = False
            output_image_embeds = last_hidden_state[output_gens_mask].view(num_imgs_for_rec, -1, dim)  # 128 x 4096 -> 2 x 64 x 4096
            recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 256 x 4096
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                if rec_loss_attr_cur == 'mse_loss':
                    # rec_loss = self.mse_loss(recon_image_embeds, target_embeds.detach())
                    mse_loss = F.mse_loss(recon_image_embeds, target_embeds.detach()) # for zero3 compatibility
                    rec_loss += rec_loss_attr_weight_cur * mse_loss
                    rec_loss_attr_value['mse_loss'] = mse_loss
                elif rec_loss_attr_cur == 'cosine_loss':
                    cosine_loss = get_cosine_loss(recon_image_embeds, target_embeds.detach())
                    rec_loss += rec_loss_attr_weight_cur * cosine_loss
                    rec_loss_attr_value['cosine_loss'] = cosine_loss
                elif rec_loss_attr_cur == 'diffusion_loss':
                    diffusion_loss = self.adapter.train_pipeline(recon_image_embeds, target_image_aug, recon_image_embeds.device, recon_image_embeds.dtype)['total_loss']
                    rec_loss += rec_loss_attr_weight_cur * diffusion_loss
                    rec_loss_attr_value['diffusion_loss'] = diffusion_loss
                else:
                    raise NotImplementedError
        else:
            output_image_embeds = torch.randn(1, self.input_resampler.num_queries,
                                              self.input_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype) + 0.0 * last_hidden_state[0, :self.input_resampler.num_queries, :]
            recon_image_embeds = self.output_resampler(output_image_embeds)

            rec_loss = 0.0 * recon_image_embeds.sum()
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                loss_value = 0.0 * recon_image_embeds.sum()
                rec_loss_attr_value[rec_loss_attr_cur] = loss_value
                if rec_loss_attr_cur in ['cosine_loss', 'diffusion_loss', 'inside_mse_loss']:
                    rec_loss += rec_loss_attr_weight_cur * loss_value
                elif rec_loss_attr_cur != 'mse_loss':
                    raise NotImplementedError

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss
        
        ret = {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'rec_loss': rec_loss,
        }
        ret.update(rec_loss_attr_value)
        # if has_image_output:
        #     ret['recon_image_embeds'] = recon_image_embeds
        #     ret['target_embeds'] = target_embeds

        return ret
    
    
class ContinuousLVLMJointLLMSDVARGenFirst(ContinuousLVLMJointLLMSDVAR):
    
    def forward(self, input_ids, attention_mask, labels, image_embeds, embeds_gen_mask, embeds_cmp_mask, ids_gen_mask,
            ids_cmp_mask, patch_positions=None, images_aug=None):
        
        assert images_aug is not None

        input_embeds = self.llm.get_input_embeddings()(input_ids)  # bz x seq_len x dim, 4 x 160 x 4096

        bz, sq, dim = input_embeds.shape

        if image_embeds is not None:
            image_embeds_cmp = image_embeds[embeds_cmp_mask]  # num_imgs_in_batch x nq_in x dim_in, 4 x 64 x 4096
            if patch_positions is not None:
                patch_positions = patch_positions[embeds_cmp_mask]
        

        if image_embeds is not None and image_embeds_cmp.shape[0] > 0:
            image_embeds_lm = self.input_resampler(image_embeds_cmp)  # num_imgs_in_batch x nq x dim, 4 x 64 x 4096
            # TODO[huangzp]: check it 
            # if self.add_patch_pos and patch_positions is not None:
            #     # assert patch_positions is not None
            #     patch_positions = patch_positions.to(
            #                                 image_embeds_lm
            #                                 ) 
            #     rel_pos_embed = torch.mm(torch.cat([patch_positions, 1-patch_positions], dim=-1)/2, self.patch_pos_embed).unsqueeze(1)
            #     image_embeds_lm = image_embeds_lm + rel_pos_embed
            has_image_cmp = True

        has_image_input = image_embeds is not None and embeds_cmp_mask.sum().item() > 0
        has_image_output = image_embeds is not None and embeds_gen_mask.sum().item() > 0
        
        

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0
        if has_image_output:
            rec_loss = 0.0
            target_embeds = image_embeds[embeds_gen_mask]
            target_embeds_lm = self.input_resampler(target_embeds) 
            if self.vit_down:
                b, n, c = target_embeds.shape
                sqrt_n = int(n**0.5)
                target_embeds = target_embeds.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
                stride = int(sqrt_n // (self.n_query ** 0.5))
                target_embeds = F.avg_pool2d(target_embeds, kernel_size=(stride, stride), stride=stride)
                target_embeds = target_embeds.view(b, c, -1).permute(0, 2, 1).contiguous()  

            with torch.no_grad():
                # get the origin mse loss
                input_embeds[ids_gen_mask] = target_embeds_lm.reshape(-1, dim)
                origin_output_lm = self.llm(attention_mask=attention_mask,
                                    inputs_embeds=input_embeds,
                                    labels=labels,
                                    output_hidden_states=True,
                                    return_dict=True)
                origin_lm_loss = origin_output_lm['loss']
                print(f"origin_lm_loss: {origin_lm_loss}")
                first_gen_ret = {
                    'origin_lm_loss': origin_lm_loss
                }
                
                
                
                first_image_index = ids_gen_mask.to(torch.int).argmax(dim=1)
                # print(f'first_image_index: {first_image_index}')
                # left to first_image_index is all True, others are False
                # the first gen input is not the same length, can not parallel generated
                # first_gen_mask = torch.zeros_like(ids_gen_mask, dtype=torch.bool)
                batch_first_gen_lm = []
                for i in range(first_image_index.shape[0]):
                    first_gen_input = input_embeds[i, :first_image_index[i], :].unsqueeze(0)
                    past_key_values = None
                    for j in range(self.n_query):
                        first_gen_output = self.llm(
                            inputs_embeds=first_gen_input,
                            past_key_values=past_key_values,
                            use_cache=True,
                            return_dict=True,
                            output_attentions=True,
                            output_hidden_states=True,
                        )
                        next_embedding = first_gen_output.hidden_states[-1][:, -1, :]
                        first_gen_input = torch.cat([first_gen_input, next_embedding.unsqueeze(1)], dim=1)
                        past_key_values = first_gen_output.past_key_values
                    # get the output for the first image
                    first_gen_lm = first_gen_input[:, - self.n_query:, :]
                    batch_first_gen_lm.append(first_gen_lm)
                first_gen_lm = torch.cat(batch_first_gen_lm, dim=0)
                print(f'first_gen_lm: {first_gen_lm.shape}')
                
                # replace the first gen output to the input_embeds
                input_embeds[ids_gen_mask] = first_gen_lm.reshape(-1, dim)
                
                # log the mse of first_gen_lm and target_embeds
                first_gen = self.output_resampler(first_gen_lm)
                print("first_gen shape: ", first_gen.shape)
                print("target_embeds shape: ", target_embeds.shape)
                first_gen_mse_loss = F.mse_loss(first_gen, target_embeds.detach())
                print("first_gen_mse_loss: ", first_gen_mse_loss)
                first_gen_ret.update({
                    'first_gen_mse_loss': first_gen_mse_loss
                })
                # 计算每个token的MSE损失
                first_gen_per_token_mse_loss = F.mse_loss(first_gen, target_embeds.detach(), reduction='none').mean(dim=(0, 2))
                for i in range(first_gen_per_token_mse_loss.shape[0]):
                    first_gen_ret[f'first_gen_per_token_mse_loss_{i}'] = first_gen_per_token_mse_loss[i]
                    print(f'first_gen_per_token_mse_loss_{i}: {first_gen_per_token_mse_loss[i].item()}')
            
        # wait for all process to finish
        torch.distributed.barrier()
        output_lm = self.llm(attention_mask=attention_mask,
                             inputs_embeds=input_embeds,
                             labels=labels,
                             output_hidden_states=True,
                             return_dict=True)
        lm_loss = output_lm['loss']
        # print(f'lm_loss: {lm_loss}')

        last_hidden_state = output_lm.hidden_states[-1]  # 4 x 160 x 4096
        
        rec_loss_attr_value = {}
        if has_image_output:
            target_image_aug = images_aug[embeds_gen_mask]
            num_imgs_for_rec = target_embeds.shape[0]
            # shift ids_gen_mask to left by 1 to get the output ids mask
            output_gens_mask = ids_gen_mask.clone()
            output_gens_mask[:, :-1] = ids_gen_mask[:, 1:]
            output_gens_mask[:, -1] = False
            output_image_embeds = last_hidden_state[output_gens_mask].view(num_imgs_for_rec, -1, dim)  # 128 x 4096 -> 2 x 64 x 4096
            recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 256 x 4096
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                if rec_loss_attr_cur == 'mse_loss':
                    # rec_loss = self.mse_loss(recon_image_embeds, target_embeds.detach())
                    mse_loss = F.mse_loss(recon_image_embeds, target_embeds.detach()) # for zero3 compatibility
                    rec_loss += rec_loss_attr_weight_cur * mse_loss
                    rec_loss_attr_value['mse_loss'] = mse_loss
                    check_first_gen_correct_mse_loss = F.mse_loss(recon_image_embeds, first_gen)
                    rec_loss_attr_value['check_first_gen_correct_mse_loss'] = check_first_gen_correct_mse_loss
                elif rec_loss_attr_cur == 'cosine_loss':
                    cosine_loss = get_cosine_loss(recon_image_embeds, target_embeds.detach())
                    rec_loss += rec_loss_attr_weight_cur * cosine_loss
                    rec_loss_attr_value['cosine_loss'] = cosine_loss
                elif rec_loss_attr_cur == 'diffusion_loss':
                    diffusion_loss = self.adapter.train_pipeline(recon_image_embeds, target_image_aug, recon_image_embeds.device, recon_image_embeds.dtype)['total_loss']
                    rec_loss += rec_loss_attr_weight_cur * diffusion_loss
                    rec_loss_attr_value['diffusion_loss'] = diffusion_loss
                else:
                    raise NotImplementedError
        else:
            output_image_embeds = torch.randn(1, self.input_resampler.num_queries,
                                              self.input_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype) + 0.0 * last_hidden_state[0, :self.input_resampler.num_queries, :]
            recon_image_embeds = self.output_resampler(output_image_embeds)

            rec_loss = 0.0 * recon_image_embeds.sum()
            for rec_loss_attr_cur, rec_loss_attr_weight_cur in zip(self.rec_loss_attr, self.rec_loss_attr_weight):
                loss_value = 0.0 * recon_image_embeds.sum()
                rec_loss_attr_value[rec_loss_attr_cur] = loss_value
                if rec_loss_attr_cur in ['cosine_loss', 'diffusion_loss', 'inside_mse_loss']:
                    rec_loss += rec_loss_attr_weight_cur * loss_value
                elif rec_loss_attr_cur != 'mse_loss':
                    raise NotImplementedError

        total_loss = self.lm_loss_scale * lm_loss + self.rec_loss_scale * rec_loss
        
        ret = {
            'total_loss': total_loss,
            'lm_loss': lm_loss,
            'rec_loss': rec_loss,
        }
        ret.update(rec_loss_attr_value)
        # if has_image_output:
        #     ret['recon_image_embeds'] = recon_image_embeds
        #     ret['target_embeds'] = target_embeds
        #     ret.update(first_gen_ret)

        return ret
    