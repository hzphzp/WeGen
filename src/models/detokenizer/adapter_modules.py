import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from typing import List
from diffusers import StableDiffusionXLPipeline
from PIL import Image
from .pipeline_stable_diffusion_xl_t2i_edit import StableDiffusionXLText2ImageAndEditPipeline


def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per
    https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr



class SDXLAdapter(nn.Module):

    def __init__(self, unet, resampler, full_ft=False, vit_down=False, snr_gamma=None, use_neg=True) -> None:
        super().__init__()
        self.unet = unet
        self.resampler = resampler
        self.full_ft = full_ft
        self.set_trainable_v2()
        self.vit_down = vit_down
        self.snr_gamma = snr_gamma
        self.use_neg = use_neg
    
    def set_trainable_v2(self):
        self.resampler.requires_grad_(True)
        adapter_parameters = []
        if self.full_ft:
            self.unet.requires_grad_(True)
            adapter_parameters.extend(self.unet.parameters())
        else:
            self.unet.requires_grad_(False)
            for name, module in self.unet.named_modules():
                if name.endswith('to_k') or name.endswith('to_v'):
                    if module is not None:
                        adapter_parameters.extend(module.parameters())
        self.adapter_parameters = adapter_parameters
        
    def freeze_unet(self):
        self.unet.requires_grad_(False)
    
    def unfreeze_unet(self):
        self.unet.requires_grad_(True)
            

    def params_to_opt(self):
        return itertools.chain(self.resampler.parameters(), self.adapter_parameters)
    
    def train_pipeline(self, image_embeds, images_aug, device, weight_dtype):
        
        # get latent and noise from images_aug (sd transform)
        latents = self.compute_vae_encodings(images_aug)['model_input']
        # add noise to latents
        bsz = latents.shape[0]
        timesteps = torch.randint(
                0, self.scheduler.config.num_train_timesteps, (bsz,), device=latents.device
            )
        timesteps = timesteps.long()
        
        noise = torch.randn_like(latents)
        noisy_encoded_pixel_values = self.scheduler.add_noise(latents, noise, timesteps)
        #  EDM Training
        #     schedule_timesteps = self.scheduler.timesteps.to(device, dtype=weight_dtype)
        #     step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        #     sigmas = self.scheduler.sigmas.to(device=device, dtype=weight_dtype)
        #     sigma = sigmas[step_indices].flatten()
        #     noisy_encoded_pixel_values = noisy_encoded_pixel_values / (
        #         (sigma[:, None, None, None] ** 2 + 1) ** 0.5
        #     )
        
        # input
        noisy_latents=noisy_encoded_pixel_values
        text_embeds = None
        time_ids=torch.zeros([bsz, 1, 6], device=latents.device, dtype=torch.long)
        # input forward
        image_embeds, pooled_image_embeds = self.resampler(image_embeds)

        unet_added_conditions = {"time_ids": time_ids, 'text_embeds': pooled_image_embeds}

        noise_pred = self.unet(noisy_latents, timesteps, image_embeds, added_cond_kwargs=unet_added_conditions).sample

        # if noise is not None:
        # loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") * loss_weight
        
        if self.snr_gamma is None:
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(
                dim=1
            )[0]
            mse_loss_weights = mse_loss_weights / snr

            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        return {'total_loss': loss, 'noise_pred': noise_pred}

    def encode_image_embeds(self, image_embeds):
        image_embeds, pooled_image_embeds = self.resampler(image_embeds)

        return image_embeds, pooled_image_embeds

    @classmethod
    def from_pretrained(cls, unet, resampler, pretrained_model_path=None, **kwargs):
        model = cls(unet=unet, resampler=resampler, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            # import pdb; pdb.set_trace()
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        return model

    def init_pipe(self,
                  vae,
                  scheduler,
                  visual_encoder,
                  image_transform,
                  discrete_model=None,
                  dtype=torch.float16,
                  device='cuda'):
        self.device = device
        self.dtype = dtype
        sdxl_pipe = StableDiffusionXLPipeline(tokenizer=None,
                                              tokenizer_2=None,
                                              text_encoder=None,
                                              text_encoder_2=None,
                                              vae=vae,
                                              unet=self.unet,
                                              scheduler=scheduler)

        self.sdxl_pipe = sdxl_pipe  #.to(self.device, dtype=self.dtype)
        # print(sdxl_pipe.text_encoder_2, sdxl_pipe.text_encoder)

        # self.visual_encoder = visual_encoder.to(self.device, dtype=self.dtype) if visual_encoder is not None else None
        self.visual_encoder = visual_encoder
        if discrete_model is not None:
            self.discrete_model = discrete_model.to(self.device, dtype=self.dtype)
        else:
            self.discrete_model = None
        self.image_transform = image_transform
        # self.vae = vae.to(self.device, dtype=self.dtype)
        self.vae = vae
        self.scheduler = scheduler

    @torch.no_grad()
    def compute_vae_encodings(self, images):
        pixel_values = images
        # pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        pixel_values = pixel_values.contiguous().float()
        pixel_values = pixel_values.float()
        pixel_values = pixel_values.to(self.vae.device, dtype=self.vae.dtype)
        with torch.no_grad():
            model_input = self.vae.encode(pixel_values).latent_dist.sample()
        model_input = model_input * self.vae.config.scaling_factor
        return {"model_input": model_input}

    @torch.no_grad()
    def get_image_embeds(self, image_pil=None, image_tensor=None, image_embeds=None, return_negative=True, image_size=448):
        return_negative = self.use_neg and return_negative
        assert int(image_pil is not None) + int(image_tensor is not None) + int(image_embeds is not None) == 1

        if image_pil is not None:
            image_tensor = self.image_transform(image_pil).unsqueeze(0).to(self.device, dtype=self.dtype)
        if image_tensor is not None:
            if return_negative:
                image_tensor_neg = torch.zeros_like(image_tensor)
                image_tensor = torch.cat([image_tensor, image_tensor_neg], dim=0)

            image_embeds = self.visual_encoder(image_tensor)
        elif return_negative:
            assert image_embeds is not None
            n_query = image_embeds.shape[1]
            image_tensor_neg = torch.zeros(1, 3, image_size, image_size).to(image_embeds.device, dtype=image_embeds.dtype)
            image_embeds_neg = self.visual_encoder(image_tensor_neg)
            if self.vit_down:
                b, n, c = image_embeds_neg.shape
                sqrt_n = int(n**0.5)
                image_embeds_neg = image_embeds_neg.permute(0, 2, 1).view(b, c, sqrt_n, sqrt_n)
                stride = int(sqrt_n // (n_query ** 0.5))
                image_embeds_neg = F.avg_pool2d(image_embeds_neg, kernel_size=(stride, stride), stride=stride)
                image_embeds_neg = image_embeds_neg.view(b, c, -1).permute(0, 2, 1).contiguous()
            image_embeds = torch.cat([image_embeds, image_embeds_neg], dim=0)

        if self.discrete_model is not None:
            image_embeds = self.discrete_model.encode_image_embeds(image_embeds)
        image_embeds, pooled_image_embeds = self.encode_image_embeds(image_embeds)

        if return_negative:
            image_embeds, image_embeds_neg = image_embeds.chunk(2)
            pooled_image_embeds, pooled_image_embeds_neg = pooled_image_embeds.chunk(2)

        else:
            image_embeds_neg = None
            pooled_image_embeds_neg = None

        return image_embeds, image_embeds_neg, pooled_image_embeds, pooled_image_embeds_neg

    def generate(self,
                 image_pil=None,
                 image_tensor=None,
                 image_embeds=None,
                 seed=42,
                 height=1024,
                 width=1024,
                 guidance_scale=6,
                 num_inference_steps=30,
                 input_image_size=448,
                 **kwargs):
        if image_pil is not None:
            assert isinstance(image_pil, Image.Image)

        image_prompt_embeds, uncond_image_prompt_embeds, pooled_image_prompt_embeds, pooled_uncond_image_prompt_embeds = self.get_image_embeds(
            image_pil=image_pil,
            image_tensor=image_tensor,
            image_embeds=image_embeds,
            return_negative=True,
            image_size=input_image_size,
        )
        # print(image_prompt_embeds.shape, pooled_image_prompt_embeds.shape)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        # print(image_prompt_embeds.shape, pooled_image_prompt_embeds.shape)
        images = self.sdxl_pipe(
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds if self.use_neg else None,
            pooled_prompt_embeds=pooled_image_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_uncond_image_prompt_embeds if self.use_neg else None,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            **kwargs,
        ).images

        return images

    
class SDXLAdapterWithLatentImage(SDXLAdapter):
    def __init__(self, unet, resampler, full_ft=False, set_trainable_late=False, vit_down=False, use_neg=True) -> None:
        nn.Module.__init__(self)
        self.unet = unet
        self.resampler = resampler
        self.full_ft = full_ft
        if not set_trainable_late:
            self.set_trainable()
        self.vit_down = vit_down
        self.use_neg = use_neg
        
    
    def set_trainable(self):
        self.resampler.requires_grad_(True)
        adapter_parameters = []
        
        in_channels = 8
        out_channels = self.unet.conv_in.out_channels
        self.unet.register_to_config(in_channels=in_channels)
        self.unet.requires_grad_(False)
        with torch.no_grad():
            new_conv_in = nn.Conv2d(in_channels, out_channels, self.unet.conv_in.kernel_size, self.unet.conv_in.stride,
                                    self.unet.conv_in.padding)

            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in
        self.unet.conv_in.requires_grad_(True)

        if self.full_ft:
            self.unet.requires_grad_(True)
            adapter_parameters.extend(self.unet.parameters())
        else:
            adapter_parameters.extend(self.unet.conv_in.parameters())
            for name, module in self.unet.named_modules():
                if name.endswith('to_k') or name.endswith('to_v'):
                    if module is not None:
                        adapter_parameters.extend(module.parameters())
        self.adapter_parameters = adapter_parameters

    @classmethod
    def from_pretrained(cls, unet, resampler, pretrained_model_path=None, set_trainable_late=False, **kwargs):
        model = cls(unet=unet, resampler=resampler, set_trainable_late=set_trainable_late, **kwargs)
        if pretrained_model_path is not None:
            ckpt = torch.load(pretrained_model_path, map_location='cpu')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print('missing keys: ', len(missing), 'unexpected keys:', len(unexpected))
        if set_trainable_late:
            model.set_trainable()
        return model
        
    def init_pipe(self,
                  vae,
                  scheduler,
                  visual_encoder,
                  image_transform,
                  dtype=torch.float16,
                  device='cuda'):
        self.device = device
        self.dtype = dtype

        sdxl_pipe = StableDiffusionXLText2ImageAndEditPipeline(
            tokenizer=None,
            tokenizer_2=None,
            text_encoder=None,
            text_encoder_2=None,
            vae=vae,
            unet=self.unet,
            scheduler=scheduler,
        )

        self.sdxl_pipe = sdxl_pipe
        self.sdxl_pipe.to(device, dtype=dtype)
        self.discrete_model = None

        self.visual_encoder = visual_encoder.to(self.device, dtype=self.dtype)
        self.image_transform = image_transform
        self.vae = vae.to(self.device, dtype=self.dtype)
        self.scheduler = scheduler
        
    def generate(self,
                 image_pil=None,
                 image_tensor=None,
                 image_embeds=None,
                 latent_image=None,
                 seed=42,
                 height=1024,
                 width=1024,
                 guidance_scale=7.5,
                 num_inference_steps=30,
                 input_image_size=448,
                 **kwargs):
        if image_pil is not None:
            assert isinstance(image_pil, Image.Image)

        image_prompt_embeds, uncond_image_prompt_embeds, pooled_image_prompt_embeds, pooled_uncond_image_prompt_embeds = self.get_image_embeds(
            image_pil=image_pil,
            image_tensor=image_tensor,
            image_embeds=image_embeds,
            return_negative=True,
            image_size=input_image_size,
        )
        # print(image_prompt_embeds.shape, pooled_image_prompt_embeds.shape)
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None

        images = self.sdxl_pipe(
            image=latent_image,
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds,
            pooled_prompt_embeds=pooled_image_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_uncond_image_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            height=height,
            width=width,
            **kwargs,
        ).images
        return images
    