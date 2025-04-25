import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LogitsProcessorList
from .generation import AutoImageTokenGenerationProcessor
from .utils import load_zero3_checkpoint
from .wegen_mllm import ContinuousLVLM
from .wegen_mllm_joint_llm_sd import ContinuousLVLMJointLLMSD


BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


def get_cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


class ContinuousLVLMJointLLMSDKaimingHead(ContinuousLVLMJointLLMSD):
    def __init__(self, km_head, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.km_head = km_head
        
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
            if self.add_patch_pos and patch_positions is not None:
                # assert patch_positions is not None
                patch_positions = patch_positions.to(
                                            image_embeds_lm
                                            ) 
                rel_pos_embed = torch.mm(torch.cat([patch_positions, 1-patch_positions], dim=-1)/2, self.patch_pos_embed).unsqueeze(1)
                image_embeds_lm = image_embeds_lm + rel_pos_embed
            has_image_cmp = True
        # else:
        #     image_embeds_cmp_fake = torch.randn(  1 , self.output_resampler.num_queries,
        #                                self.output_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype)

        #     image_embeds_lm = self.input_resampler(image_embeds_cmp_fake)
        #     if self.add_patch_pos:
        #         rel_pos_embed = self.patch_pos_embed.mean(0, keepdim=True).unsqueeze(1) # 1, 1, dim
        #         image_embeds_lm = image_embeds_lm + rel_pos_embed

        #     has_image_cmp = False

        has_image_input = image_embeds is not None and embeds_cmp_mask.sum().item() > 0
        has_image_output = image_embeds is not None and embeds_gen_mask.sum().item() > 0

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0
        # TODO[huangzp]: check this
        # else:
        #     input_embeds[:1, :self.input_resampler.num_queries, :] += 0.0 * image_embeds_lm[:1, :, :]
            
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
            rec_loss = 0.0
            target_embeds = image_embeds[embeds_gen_mask]  # num_imgs_gen_target x nq_in x dim_in, 2 x 256 x 4096
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
            output_image_embeds = last_hidden_state[output_gens_mask].view(num_imgs_for_rec, -1, dim)  # 128 x 4096 -> 2 x 64 x 1792
            recon_image_embeds = self.output_resampler(output_image_embeds)  # 2 x 64 x 1792
            
            # main_precess
            # kaiming diffusion head
            # print(recon_image_embeds.shape, target_embeds.shape)
            kaiming_loss = self.km_head(z=recon_image_embeds, target=target_embeds.detach())
            rec_loss += kaiming_loss
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

        return ret

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        # return {
        #     'text': generate_text,
        #     'has_img_output': has_img_output,
        #     'img_gen_feat': img_gen_feat,
        #     'num_gen_imgs': num_gen_imgs
        # }
        if result['has_img_output']:
            img_gen_feat = result['img_gen_feat']
            b, n, c = img_gen_feat.shape
            img_gen_feat_flatten = img_gen_feat.view(-1, img_gen_feat.size(-1))
            img_gen_feat_flatten = self.km_head.sample(z=img_gen_feat_flatten, temperature=1.0, cfg=4.0)
            img_gen_feat = img_gen_feat_flatten.view(b, n, c)
            result['img_gen_feat'] = img_gen_feat
        return result
            
