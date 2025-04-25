import hydra
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
except:
    print('use NVIDIA GPU')
import os
import re
import pickle 
import argparse
import pyrootutils
from PIL import Image
from tqdm import tqdm 
from pathlib import Path
from omegaconf import OmegaConf
from accelerate import Accelerator
from torch.utils.data import Dataset
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, Transformer2DModel
# from any_res import process_anyres_image

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)


class TextDataset(Dataset):
    def __init__(self, anno_path):
        if anno_path.endswith(".txt"):
            self.all_prompts = []
            with open(anno_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue 
                    else:
                        self.all_prompts.append(line)
        else:
            self.all_prompts = pickle.load(open(anno_path, "rb"))
    
        self.all_indices = list(range(len(self.all_prompts)))

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        if prompt == None:
            prompt = ""
        return prompt 


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--validation_result_path', type=str)
parser.add_argument('--total_eval_samples', type=int)
parser.add_argument('--guidance_scale', type=float)
parser.add_argument('--var_cfg_guidance_scale', type=float, default=1.0)
args = parser.parse_args()
ckpt_path = args.ckpt_path

trained_ckpt_path = ckpt_path
agent_cfg_path = "configs/clm_models/agent_wegen_mllm.yaml"
llm_cfg_path = "configs/clm_models/llm_wegen_mllm.yaml"

# init the basic configs
BOI_TOKEN = '<img>'
BOP_TOKEN = '<patch>'
EOI_TOKEN = '</img>'
EOP_TOKEN = '</patch>'
IMG_TOKEN = '<img_{:05d}>'


dtype = torch.float16
dtype_str = 'fp16'
num_img_in_tokens = 64
num_img_out_tokens = 64
instruction_prompt = '[INST] Generate a painting: {instruction} [/INST]\n'

# some configs
tokenizer_cfg_path = 'configs/tokenizer/clm_llama_tokenizer_224loc_anyres.yaml'
image_transform_cfg_path = 'configs/processer/clip_l_448_transform_resize.yaml'
visual_encoder_cfg_path = 'configs/visual_encoder/eva_vision.yaml'
adapter_cfg_path = 'configs/sdxl_adapter/sdxl_evaclip_vit_avgpool_q64_fullft_minisnr5_freeze_unet.yaml'
discrete_model_cfg_path = 'configs/discrete_model/discrete_identity.yaml'
diffusion_model_path = './wegen_mllm_ckpt/pretrained/stable-diffusion-xl-base-1.0'
diffusion_specific_unet_path = './wegen_mllm_ckpt/stage1_final/unet'   
guidance_scale = args.guidance_scale


# visual_encoder, agent, adapter, tokenizer
# BIO_TOKEN, IMG_TOKEN, EOI_TOKEN, num_img_in_tokens, num_img_out_tokens, instruction_prompt, device
def gen(prompt_list):
    if not isinstance(prompt_list, list):
        prompt_list = [prompt_list]
    text_prompt = ''
    embeds_cmp_mask = []
    has_text = False
    for x in prompt_list:
        if isinstance(x, str):
            has_text = True
            text_prompt += x

        
    text_prompt = instruction_prompt.format_map({'instruction': text_prompt})
    print(text_prompt)

    input_ids = tokenizer.encode(text_prompt, add_special_tokens=False)
    input_ids = [tokenizer.bos_token_id] + input_ids

    input_ids = torch.tensor(input_ids).to(accelerator.device, dtype=torch.long)

        

    input_ids = input_ids.unsqueeze(0)

    with torch.no_grad():
        output = agent_model.generate(tokenizer=tokenizer,
                                    input_ids=input_ids,
                                    patch_positions=None,
                                    max_new_tokens=512,
                                    num_img_gen_tokens=num_img_out_tokens,
                                    var_cfg_guidance_scale=args.var_cfg_guidance_scale)
    output_text = re.sub('<[^>]*>', '', output['text'])

    if output['has_img_output']:
        images = adapter.generate(image_embeds=output['img_gen_feat'], num_inference_steps=50, guidance_scale=guidance_scale)
        output_image = images[0]
    else:
        output_image = None
    torch.cuda.empty_cache()
    return output_text, output_image


if __name__ == '__main__':
    accelerator_project_config = ProjectConfiguration(logging_dir=os.getcwd())
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision="no",
        project_config=accelerator_project_config,
    )

    tokenizer_cfg = OmegaConf.load(tokenizer_cfg_path)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    image_transform_cfg = OmegaConf.load(image_transform_cfg_path)
    image_transform = hydra.utils.instantiate(image_transform_cfg)

    visual_encoder_cfg = OmegaConf.load(visual_encoder_cfg_path)
    visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
    visual_encoder.eval().to(accelerator.device, dtype=dtype)
    print('Init visual encoder done')

    llm_cfg = OmegaConf.load(llm_cfg_path)
    llm = hydra.utils.instantiate(llm_cfg, torch_dtype=dtype)
    print('Init llm done.')

    noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
    print('init vae')
    vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae").to(accelerator.device, dtype=dtype)
    print('init unet')
    if diffusion_specific_unet_path is not None:
        unet = UNet2DConditionModel.from_pretrained(diffusion_specific_unet_path, use_safetensors=True, variant="bf16").to(accelerator.device, dtype=dtype)
    else:
        unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet").to(accelerator.device, dtype=dtype)

    adapter_cfg = OmegaConf.load(adapter_cfg_path)
    adapter = hydra.utils.instantiate(adapter_cfg, unet=unet).to(accelerator.device, dtype=dtype).eval()

    discrete_model_cfg = OmegaConf.load(discrete_model_cfg_path)
    discrete_model = hydra.utils.instantiate(discrete_model_cfg).to(accelerator.device).eval()
    print('Init adapter done')

    adapter.init_pipe(vae=vae,
                    scheduler=noise_scheduler,
                    visual_encoder=visual_encoder,
                    image_transform=image_transform,
                    dtype=dtype,
                    device=accelerator.device)

    print('Init adapter pipe done')

    agent_model_cfg = OmegaConf.load(agent_cfg_path)
    agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm, adapter=adapter)

    agent_model.eval().to(accelerator.device, dtype=dtype)
    print('Init agent mdoel Done')


    cktp = torch.load(trained_ckpt_path, map_location='cpu')
    missing_keys, unexpected_keys = agent_model.load_state_dict(cktp, strict=False)
    with open('missing_keys.txt', 'w') as f:
        f.write(str(missing_keys))
    with open('unexpected_keys.txt', 'w') as f:
        f.write(str(unexpected_keys))
        
        
    boi_token_id = tokenizer.encode(BOI_TOKEN, add_special_tokens=False)[0]
    eoi_token_id = tokenizer.encode(EOI_TOKEN, add_special_tokens=False)[0]

    bop_token_id = tokenizer.encode(BOP_TOKEN, add_special_tokens=False)[0]
    eop_token_id = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

    eval_batch_size = 1
    count = 0
    anno_path = 'path_to_coco_test_captions'
    img_path = os.path.join(args.validation_result_path, "images")
    txt_path = os.path.join(args.validation_result_path, f"log_{accelerator.process_index}.txt")
    if accelerator.is_main_process:
        os.makedirs(img_path, exist_ok=True)

    caption_dataset = TextDataset(anno_path)
    caption_dataloader = torch.utils.data.DataLoader(
        caption_dataset, batch_size=eval_batch_size, 
        shuffle=False, drop_last=False, num_workers=0
    ) 
    caption_dataloader = accelerator.prepare(caption_dataloader)
    
    for index, batch_prompts in tqdm(enumerate(caption_dataloader), 
        disable=not accelerator.is_main_process, 
        total=args.total_eval_samples // eval_batch_size // accelerator.num_processes):

        name = str(index * accelerator.num_processes + accelerator.process_index).zfill(6)
        loaded_prompt_list = []
        loaded_prompt_list.append(batch_prompts[0])
        # print(name)
        # print(loaded_prompt_list)
        output_text, output_image = gen(loaded_prompt_list)
        save_path = f'{img_path}/{name}'
        output_text = name + ": " + batch_prompts[0] + "\n"
        while True:
            try:
                if output_text is not None:
                    with open(txt_path, 'a') as f:
                        f.write(output_text)
                break
            except:
                print("Try again", flush=True)
        #     print(f'Saved text to {save_path}.txt')
        # if output_image is not None:
        while True:
            try:
                output_image.save(save_path+'.jpg')
                break
            except:
                print("Try again", flush=True)
        # print(f'Saved image to {save_path}.jpg')
        
        count += eval_batch_size * accelerator.num_processes
        print(count, flush=True)
        if count >= args.total_eval_samples:
            break
        accelerator.wait_for_everyone()

        
            
        
    