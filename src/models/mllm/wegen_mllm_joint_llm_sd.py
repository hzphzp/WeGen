import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LogitsProcessorList
from .generation import AutoImageTokenGenerationProcessor
from .utils import load_zero3_checkpoint
from .wegen_mllm import ContinuousLVLM
from einops import rearrange, repeat


BOI_TOKEN = '<img>'
EOI_TOKEN = '</img>'
IMG_TOKEN = '<img_{:05d}>'


def get_cosine_loss(rec, target):
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


def get_gmm_loss(embed, target, k_gaussion):
    b, f, c = embed.shape
    dim = int((c / k_gaussion - 1) / 2)
    m = embed[:, :, : (dim * k_gaussion)].view(b, f, k_gaussion, dim).to(torch.float32)  # torch.Size([8, 64, 4, 1792])
    s = embed[:, :, (dim * k_gaussion): (2 * dim * k_gaussion)].view(b, f, k_gaussion, dim).to(torch.float32)  # torch.Size([8, 64, 4, 1792])
    pi = embed[:, :, (2 * dim * k_gaussion):].to(torch.float32)  # torch.Size([8, 64, 4])
    s = torch.clamp(F.softplus(s), min=1e-5)
    pi = F.softmax(pi, dim=-1)
    # print(s, "\n\n\n\n\n", flush=True)
    # print("target: ", target, "\n\n\n\n\n", flush=True)
    # print(m, s, pi, "\n\n\n\n\n", flush=True)

    # exponent = -0.5 * torch.sum(((target - m) / s)**2, dim=-1, keepdim=False)  # torch.Size([8, 64, 4]
    # # print("exponent: ", exponent, "\n\n\n\n\n", flush=True)
    # # exp_term = torch.exp(exponent)  # torch.Size([8, 64, 4]
    # # print("exp_term: ", exp_term, "\n\n\n\n\n", flush=True)
    # denominator = torch.prod(s, dim=-1, keepdim=False)  # torch.Size([8, 64, 4]
    # # print("denominator: ", denominator, "\n\n\n\n\n", flush=True)
    # # N = (exp_term / denominator).squeeze(dim=-1)  # torch.Size([8, 64, 4])
    # N = (1 / denominator).squeeze(dim=-1)  # torch.Size([8, 64, 4])
    # # print("N: ", N, "\n\n\n\n\n", flush=True)
    # # exit()

    # p = torch.sum(N * pi, dim=-1, keepdim=False)
    # gmm_loss = (-1 * (torch.log(p) + torch.sum(exponent, dim=-1, keepdim=False)) / dim).mean()

    # blend_weight = torch.distributions.Categorical(pi)


    # return gmm_loss

    blend_weight = torch.distributions.Categorical(pi)
    comp = torch.distributions.Independent(torch.distributions.Normal(m, s), 1)
    gmm = torch.distributions.MixtureSameFamily(blend_weight, comp)
    return -(gmm.log_prob(target.to(torch.float32))).mean()


def gmm_sample(embed, k_gaussion, num_samples=1000):
    b, f, c = embed.shape
    dim = int((c / k_gaussion - 1) / 2)
    m = embed[:, :, : (dim * k_gaussion)].view(b, f, k_gaussion, dim).to(torch.float32)  # torch.Size([8, 64, 4, 1792])
    s = embed[:, :, (dim * k_gaussion): (2 * dim * k_gaussion)].view(b, f, k_gaussion, dim).to(torch.float32)  # torch.Size([8, 64, 4, 1792])
    pi = embed[:, :, (2 * dim * k_gaussion):].to(torch.float32)  # torch.Size([8, 64, 4])
    s = torch.clamp(F.softplus(s), min=1e-5)
    pi = F.softmax(pi, dim=-1)

    blend_weight = torch.distributions.Categorical(pi)
    comp = torch.distributions.Independent(torch.distributions.Normal(m, s), 1)
    gmm = torch.distributions.MixtureSameFamily(blend_weight, comp)
    samples = gmm.sample((num_samples,))

    log_probs = gmm.log_prob(samples)
    probs = torch.exp(log_probs).permute(1, 2, 0)

    # 找到概率密度最大的样本
    max_prob, max_prob_index = torch.max(probs, dim=-1)
    max_prob_index = max_prob_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, dim)
    samples = samples.permute(1, 2, 0, 3)
    max_prob_sample = torch.gather(samples, dim=2, index=max_prob_index).squeeze(dim=2)

    return max_prob_sample.to(embed.dtype)


def greedy_gmm_sample(embed, k_gaussion, num_samples=1000):
    b, f, c = embed.shape
    dim = int((c / k_gaussion - 1) / 2)
    # m = embed[:, :, : (dim * k_gaussion)].view(b, f, k_gaussion, dim).to(torch.float32)  # torch.Size([8, 64, 4, 1792])
    m = rearrange(embed[:, :, : (dim * k_gaussion)], 'b f (k d) -> b (f d) k 1', k=k_gaussion).to(torch.float32)
    # s = embed[:, :, (dim * k_gaussion): (2 * dim * k_gaussion)].view(b, f, k_gaussion, dim).to(torch.float32)  # torch.Size([8, 64, 4, 1792])
    s = rearrange(embed[:, :, (dim * k_gaussion): (2 * dim * k_gaussion)], 'b f (k d) -> b (f d) k 1', k=k_gaussion).to(torch.float32)
    # pi = embed[:, :, (2 * dim * k_gaussion):].to(torch.float32)  # torch.Size([8, 64, 4])
    pi = repeat(embed[:, :, (2 * dim * k_gaussion):], 'b f k -> b (f d) k', d=dim).to(torch.float32)
    s = torch.clamp(F.softplus(s), min=1e-5)
    pi = F.softmax(pi, dim=-1)

    blend_weight = torch.distributions.Categorical(pi)
    comp = torch.distributions.Independent(torch.distributions.Normal(m, s), 1)
    gmm = torch.distributions.MixtureSameFamily(blend_weight, comp)
    samples = gmm.sample((num_samples,))

    log_probs = gmm.log_prob(samples)
    probs = torch.exp(log_probs).permute(1, 2, 0)

    # 找到概率密度最大的样本
    max_prob, max_prob_index = torch.max(probs, dim=-1)
    max_prob_index = max_prob_index.unsqueeze(-1).unsqueeze(-1)
    samples = samples.permute(1, 2, 0, 3)
    max_prob_sample = torch.gather(samples, dim=2, index=max_prob_index).squeeze(dim=2)
    max_prob_sample = rearrange(max_prob_sample, 'b (f d) 1 -> b f d', d=dim)

    return max_prob_sample.to(embed.dtype)


class ContinuousLVLMJointLLMSD(ContinuousLVLM):

    def __init__(self, adapter, *args, **kwargs) -> None:
        # lm_loss_scale: 1.0
        # # rec_loss_scale: 6.0
        # mse_rec: True
        # mse_rec_loss_scale: 6.0
        # diffusion_rec_loss_scale: 1.0
        self.rec_loss_attr = kwargs.pop('rec_loss_attr', [])
        self.rec_loss_attr_weight = kwargs.pop('rec_loss_attr_weight', [])

        output_proj_in_features = kwargs.pop('output_proj_in_features', None)
        output_proj_out_features = kwargs.pop('output_proj_out_features', None)
        output_proj_bias = kwargs.pop('output_proj_bias', False)

        assert len(self.rec_loss_attr) == len(self.rec_loss_attr_weight)
        super().__init__(*args, **kwargs)

        if "gmm_loss" in self.rec_loss_attr:
            assert output_proj_in_features is not None and output_proj_out_features is not None, \
            "!!!!!! use gmm loss, must have output_proj layer !!!!!!"
            self.k_gaussion = int(output_proj_out_features / (2 * output_proj_in_features + 1))
            self.output_proj = nn.Linear(in_features=output_proj_in_features, 
                                         out_features=output_proj_out_features,
                                         bias=output_proj_bias
                                         )

        print("init the adapter successfully")
        if 'diffusion_loss' in self.rec_loss_attr:
            self.adapter = adapter
        else:
            self.adapter = None

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
        else:
            image_embeds_cmp_fake = torch.randn(  1 , self.output_resampler.num_queries,
                                       self.output_resampler.embed_dim).to(input_embeds.device, dtype=input_embeds.dtype)

            image_embeds_lm = self.input_resampler(image_embeds_cmp_fake)
            if self.add_patch_pos:
                rel_pos_embed = self.patch_pos_embed.mean(0, keepdim=True).unsqueeze(1) # 1, 1, dim
                image_embeds_lm = image_embeds_lm + rel_pos_embed

            has_image_cmp = False

        has_image_input = image_embeds is not None and embeds_cmp_mask.sum().item() > 0
        has_image_output = image_embeds is not None and embeds_gen_mask.sum().item() > 0

        if has_image_input:
            input_embeds[ids_cmp_mask] = image_embeds_lm.reshape(-1, dim)  # eg, 128 x 4096
            # zero_loss = 0.0
        # TODO[huangzp]: check this
        else:
            input_embeds[:1, :self.input_resampler.num_queries, :] += 0.0 * image_embeds_lm[:1, :, :]
            
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
            output_image_embeds = last_hidden_state[output_gens_mask].view(num_imgs_for_rec, -1, dim)
            recon_image_embeds = self.output_resampler(output_image_embeds)
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
                elif rec_loss_attr_cur == 'gmm_loss':
                    gmm_embeds = self.output_proj(recon_image_embeds)
                    gmm_loss = get_gmm_loss(gmm_embeds, target_embeds.detach(), self.k_gaussion)
                    rec_loss += rec_loss_attr_weight_cur * gmm_loss
                    rec_loss_attr_value['gmm_loss'] = gmm_loss
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

    def generate_gmm(self,
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
                 patch_positions=None):
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

        generation_config = {
            'temperature': temperature,
            'num_beams': num_beams,
            'max_new_tokens': max_new_tokens,
            'top_p': top_p,
            'do_sample': False
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

            if "gmm_loss" in self.rec_loss_attr:
                img_gen_feat = self.output_proj(img_gen_feat)
                img_gen_feat = greedy_gmm_sample(img_gen_feat, self.k_gaussion)

        else:
            img_gen_feat = None

        text_mask[generate_ids == boi_token_id] = False
        generate_ids = generate_ids[text_mask]
        generate_text = tokenizer.decode(generate_ids, skip_special_tokens=False)

        return {
            'text': generate_text,
            'has_img_output': has_img_output,
            'img_gen_feat': img_gen_feat,
            'num_gen_imgs': num_gen_imgs
        }

    @classmethod
    def from_pretrained(cls, llm, input_resampler, output_resampler, pretrained_model_path=None, **kwargs):
        model = cls(llm=llm, input_resampler=input_resampler, output_resampler=output_resampler, **kwargs)
        if os.environ.get('DEBUG_FLAG', 'False') == 'True':
            return model

        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            load_zero3_checkpoint(model, ckpt)
        return model
