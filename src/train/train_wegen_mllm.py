from datetime import timedelta
import hydra
import pyrootutils
import os
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
except:
    print('use NVIDIA GPU')
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
import argparse
from typing import Optional
import transformers
from dataclasses import dataclass, field, asdict, is_dataclass
from torchdata.dataloader2 import DataLoader2, MultiProcessingReadingService, DistributedReadingService, SequentialReadingService
import gc
import logging
import glob

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import torchvision
from torchvision import datasets, transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, EulerDiscreteScheduler, DDPMScheduler

print('============= train code')

pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True)
from src.train.schedular import get_scheduler
from src.train.dist_utils import all_gather

# logger = get_logger(__name__, log_level='info')
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

logger = logging.getLogger(__name__)
# os.environ["WANDB_MODE"] = "offline"


@dataclass
class ConfigPathArguments:
    image_transform: Optional[str] = field(default=None, metadata={"help": "config path of image transform (CLIP model)"})
    image_transform2: Optional[str] = field(default=None, metadata={"help": "config path of image transform augmentation, (SDXL)"})
    tokenizer: Optional[str] = field(default=None, metadata={"help": "config path of tokenizer used to initialize tokenizer"})
    # model: Optional[str] = field(default=None, metadata={"help": "config path of llm"})
    visual_encoder: Optional[str] = field(default=None, metadata={"help": "config path of visual encoder"})
    adapter_cfg_path: Optional[str] = field(default=None, metadata={"help": "config path of adapter"})
    diffusion_model_path: Optional[str] = field(default=None, metadata={"help": "config path of sdxl weight"})
    diffusion_specific_unet_path: Optional[str] = field(default=None, metadata={"help": "config path of specific sdxl unet weight"})
    llm_model: Optional[str] = field(default=None, metadata={"help": "config path of llm"})
    agent_model: Optional[str] = field(default=None, metadata={"help": "config path of agent"})
    train_dataset: Optional[str] = field(default=None, metadata={"help": "config path of training dataset"})
    fsdp_plugin: Optional[str] = field(default=None, metadata={"help": "config path of fsdp plugin"})
    deepspeed_plugin: Optional[str] = field(default=None, metadata={"help": "config path of deepspeed plugin"})


@dataclass
class TrainingArguments:
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}, )
    validation_path: str = field(
        default='./validation', metadata={"help": "The path to the validation images."})
    training_sample_steps: int = field(default=1000, metadata={"help": "Number of updates steps before the training sample."})
    resume_from_checkpoint: Optional[str] = field(
        default=None, metadata={"help": "The path to a folder with a valid checkpoint for your model."})
    auto_resume: bool = field(default=False, metadata={"help": "Whether to automatically resume from the latest checkpoint in the output directory."})
    resume_steps: Optional[int] = field(default=None, metadata={"help": "The training steps of saved checkpoint"})
    eval_steps: Optional[int] = field(default=1000, metadata={"help": "Number of updates steps before the evaluation."})
    batch_size: Optional[int] = field(default=60, metadata={"help": "The training batch size"})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    proj_lr_scale: float = field(default=1., metadata={"help": "The learning rate scale for project."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."})
    mixed_precision: Optional[str] = field(
        default='no',
        metadata={
            "help":
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >=1.10.and an Nvidia Ampere GPU."
        })
    num_train_epochs: int = field(default=3, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(default=-1, metadata={"help": "Total number of training steps to perform. "})
    save_steps: int = field(default=10000, metadata={"help": "Number of updates steps before two checkpoint saves."})
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The scheduler type to use."})
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    min_lr_ratio: float = field(default=0.01, metadata={"help": "Minimal learning rate ratio."})
    dataloader_num_workers: int = field(default=8, metadata={"help": "The number of workers to use for data loading."})
    project_name: str = field(default="ContinuousVLM", metadata={"help": "The name of experiment"})
    expr_name: str = field(default="", metadata={"help": "The name of experiment"})


def build_dataloader(dataset_cfg, image_transform, tokenizer, batch_size, dataloader_num_workers=4, *args, **kwargs):
    print("build dataset")
    dataset = hydra.utils.instantiate(dataset_cfg, image_transform=image_transform, tokenizer=tokenizer, *args, **kwargs)
    print("dataset build done")
    mp_service = MultiProcessingReadingService(num_workers=dataloader_num_workers)
    dist_service = DistributedReadingService(timeout=3600)
    reading_service = SequentialReadingService(dist_service, mp_service)
    dataloader = DataLoader2(dataset, reading_service=reading_service)
    # dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=dataloader_num_workers)
    return dataloader


def get_metric(output):
    metric = {}
    for key, value in output.items():
        if 'loss' in key:
            gathered_metric = torch.stack(all_gather(value)).mean()
            # metric[key] = value.item()
            metric[key] = gathered_metric.item()
        if 'acc' in key:
            metric[key] = value.item()
    return metric


def merge_config(**kwargs):
    config = {}
    for key, value in kwargs.items():
        if isinstance(value, argparse.Namespace):
            config[key] = vars(value)
        elif isinstance(value, DictConfig):
            config[key] = OmegaConf.to_object(value)
        elif is_dataclass(value):
            config[key] = asdict(value)
        elif isinstance(value, (int, str, float, dict)) or value is None:
            config[key] = value
        else:
            logger.error(f'key: {key}, value: {value} will not be merged.')
    return config


def trainable_params(model):
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            count += param.numel()
    return count


def train():

    parser = transformers.HfArgumentParser((ConfigPathArguments, TrainingArguments))
    cfg_path, args = parser.parse_args_into_dataclasses()

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, 'logs'))

    assert int(cfg_path.fsdp_plugin is not None) + int(cfg_path.deepspeed_plugin is not None) <= 1
    if cfg_path.fsdp_plugin is not None:
        fsdp_plugin_cfg = OmegaConf.load(cfg_path.fsdp_plugin)
        fsdp_plugin = hydra.utils.instantiate(fsdp_plugin_cfg)
        logger.info('Use FSDP plugin')
    else:
        fsdp_plugin = None

    if cfg_path.deepspeed_plugin is not None:
        deepspeed_plugin_cfg = OmegaConf.load(cfg_path.deepspeed_plugin)
        deepspeed_plugin = hydra.utils.instantiate(deepspeed_plugin_cfg)
        logger.info('Use deepspeed plugin')
    else:
        deepspeed_plugin = None
        
    print("-*-" * 20, "start init accelerator", flush=True)
    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=['tensorboard'],
        # log_with=['tensorboard', 'wandb'] if os.environ.get('DEBUG_FLAG', 'False') != 'True' else ['tensorboard'],
        project_config=project_config,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        step_scheduler_with_optimizer=False,
        fsdp_plugin=fsdp_plugin,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[InitProcessGroupKwargs(
            timeout=timedelta(seconds=3600))],
    )
    accelerator.wait_for_everyone()
    logger.info('Init accelerator done.')

    if cfg_path.deepspeed_plugin is not None:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 8


    os.makedirs(args.output_dir, exist_ok=True)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    image_transform_cfg = OmegaConf.load(cfg_path.image_transform)
    image_transform = hydra.utils.instantiate(image_transform_cfg)
    image_transform2_cfg = OmegaConf.load(cfg_path.image_transform2)
    image_transform2 = hydra.utils.instantiate(image_transform2_cfg)

    tokenizer_cfg = OmegaConf.load(cfg_path.tokenizer)
    tokenizer = hydra.utils.instantiate(tokenizer_cfg)

    train_dataset_cfg = OmegaConf.load(cfg_path.train_dataset)

    visual_encoder_cfg = OmegaConf.load(cfg_path.visual_encoder)
    visual_encoder = hydra.utils.instantiate(visual_encoder_cfg)
    logger.info('Load visual encoder done.')

    llm_model_cfg = OmegaConf.load(cfg_path.llm_model)
    llm_model = hydra.utils.instantiate(llm_model_cfg, torch_dtype=weight_dtype)
    llm_model.gradient_checkpointing_enable()
    llm_model.config.use_cache = False
    logger.info('Load llm model done.')
    
    
    # sd init
    
    adapter_cfg_path = cfg_path.adapter_cfg_path
    adapter_cfg = OmegaConf.load(adapter_cfg_path)
    diffusion_model_path = cfg_path.diffusion_model_path

    logger.info('init vae')
    vae = AutoencoderKL.from_pretrained(diffusion_model_path, subfolder="vae")
    logger.info('init noise scheduler')
    # noise_scheduler = EulerDiscreteScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
    noise_scheduler = DDPMScheduler.from_pretrained(diffusion_model_path, subfolder="scheduler")
    logger.info('init unet')
    if cfg_path.diffusion_specific_unet_path is not None:
        unet = UNet2DConditionModel.from_pretrained(cfg_path.diffusion_specific_unet_path, use_safetensors=True, variant="bf16")
    else:
        unet = UNet2DConditionModel.from_pretrained(diffusion_model_path, subfolder="unet")
    unet.enable_gradient_checkpointing()
    
    visual_encoder.to(accelerator.device, dtype=weight_dtype)
    logger.info('Freeze visual encoder...')
    visual_encoder.requires_grad_(False)
    visual_encoder.eval()
    
    logger.info('Freeze visual vae...')
    vae.requires_grad_(False)
    vae = vae.eval()
    
    logger.info('init sdxl adapter')
    adapter = hydra.utils.instantiate(adapter_cfg, unet=unet)

    adapter.init_pipe(vae=vae,
                    scheduler=noise_scheduler,
                    visual_encoder=None,  # TODO[huangzp]: check here
                    image_transform=image_transform,
                    device=accelerator.device, 
                    dtype=weight_dtype
                    )

    agent_model_cfg = OmegaConf.load(cfg_path.agent_model)
    agent_model = hydra.utils.instantiate(agent_model_cfg, llm=llm_model, adapter=adapter)
    logger.info('Load agent model done.')


    if cfg_path.fsdp_plugin is not None:
        agent_model = accelerator.prepare(agent_model)
        
    # TODO[huangzp]: bug here in the following code
    # scale up the learning rate for the agent.input_resampler and agent.output_resampler
    base_params = [param for name, param in agent_model.named_parameters() if 'input_resampler' not in name and 'output_resampler' not in name]
    proj_params = [param for name, param in agent_model.named_parameters() if 'input_resampler' in name or 'output_resampler' in name]
    

    optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': args.learning_rate},
        {'params': proj_params, 'lr': args.learning_rate * args.proj_lr_scale}
        ],
                                  betas=[args.adam_beta1, args.adam_beta2],
                                  eps=args.adam_epsilon,
                                  weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(agent_model.parameters(),
    #                               lr=args.learning_rate,
    #                               betas=[args.adam_beta1, args.adam_beta2],
    #                               eps=args.adam_epsilon,
    #                               weight_decay=args.weight_decay)
    
    logger.info('Init optimizer done.')
    scheduler = get_scheduler(name=args.lr_scheduler_type,
                              optimizer=optimizer,
                              num_warmup_steps=args.warmup_steps,
                              num_training_steps=args.max_steps,
                              min_lr_ratio=args.min_lr_ratio)
    # accelerator.register_for_checkpointing(scheduler)
    print("build dataloader")
    train_dataloader = build_dataloader(dataset_cfg=train_dataset_cfg,
                                        image_transform=image_transform,
                                        image_transform2=image_transform2,
                                        tokenizer=tokenizer,
                                        batch_size=args.batch_size,
                                        dataloader_num_workers=args.dataloader_num_workers)
    print("dataloader build done")
    accelerator.wait_for_everyone()
    if cfg_path.fsdp_plugin is not None:
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)
    else:
        agent_model, optimizer, scheduler = accelerator.prepare(agent_model, optimizer, scheduler)
        # adapter all ready prepared in agent_model
    logger.info('Prepare accelerator done.')

    config_record = merge_config(agent_model=agent_model_cfg,
                                 llm_model=llm_model_cfg,
                                 adapter=adapter_cfg,
                                 visual_encoder=visual_encoder_cfg,
                                 image_transform=image_transform_cfg,
                                 tokenizer=tokenizer_cfg,
                                 train_dataset=train_dataset_cfg,
                                 train_args=args)
    accelerator.init_trackers(project_name="wegen_mllm",
                              init_kwargs={"wandb": {
                                  "config": config_record,
                                  "name": args.expr_name,
                                  "dir": args.output_dir,
                                #   "project": 'MultiModal-LLM Research', 
                                  "entity": 'hzp1104',
                                  "resume": 'allow',
                                  'mode': 'offline'
                              }})
    global_step = 0
    if args.auto_resume:
        print('auto resume')
        # find the latest checkpoint in args.output_dir
        checkpoint_files = glob.glob(os.path.join(args.output_dir, 'checkpoint-*'))
        if len(checkpoint_files) != 0:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[-1]))
            print(f'latest_checkpoint: {latest_checkpoint}')
            args.resume_steps = int(latest_checkpoint.split('-')[-1])
            args.resume_from_checkpoint = latest_checkpoint
        else:
            print('no autoresume checkpoint found')
    if args.resume_from_checkpoint is not None:
        # if args.resume_from_checkpoint is dir
        if os.path.isdir(args.resume_from_checkpoint):
            logger.info(f'Load checkpoint from {args.resume_from_checkpoint}')
            accelerator.load_state(args.resume_from_checkpoint)
            # train_dataloader.resume()
            print(f'global_step: {global_step}')
        # end with pytorch_model.bin
        elif args.resume_from_checkpoint.endswith('pytorch_model.bin'):
            cktp = torch.load(args.resume_from_checkpoint, map_location='cpu')
            # unwarp model to load ckpt
            missing_keys, unexpected_keys = accelerator.unwrap_model(agent_model).load_state_dict(cktp)
            with open('resume_missing_keys.txt', 'w') as f:
                f.write(str(missing_keys))
            with open('resume_unexpected_keys.txt', 'w') as f:
                f.write(str(unexpected_keys))
            print(f"load from {args.resume_from_checkpoint} done. missing_keys: resume_missing_keys.txt, unexpected_keys: resume_unexpected_keys.txt")

        gc.collect()

    num_params = trainable_params(agent_model)
    logger.info("***** Running training *****")
    logger.info(f"  Total optimization steps = {args.max_steps}")
    logger.info(f"  Total trainable params = {num_params}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Steps")
    if args.resume_steps is not None:
        global_step = args.resume_steps
        progress_bar.update(args.resume_steps)

    for epoch in range(args.num_train_epochs):
        # adapter.train()
        agent_model.train()
        logger.info('Start new epoch')

        #  change seed 
        if args.resume_steps is not None:
            seed = args.resume_steps + epoch + 42
        else:
            seed = epoch + 42
        train_dataloader.seed(seed)
        should_log_sample = False  
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(agent_model):
                images = batch['images'].to(accelerator.device, dtype=weight_dtype) if batch['images'] is not None else None
                image_embeds = None
                # print(f"current_rank: {accelerator.process_index}, step: {step}, images: {images.shape}, caller: {batch['caller']}")
                if images is not None:
                    assert images.shape[0] != 0
                    with torch.no_grad():
                        image_embeds = visual_encoder(images) 
                output = agent_model(input_ids=batch['input_ids'].to(accelerator.device),
                                     attention_mask=batch['attention_mask'].to(accelerator.device),
                                     labels=batch['labels'].to(accelerator.device),
                                     image_embeds=image_embeds,
                                     patch_positions=batch['patch_position'].to(accelerator.device) if 'patch_position' in batch else None,
                                     embeds_gen_mask=batch['embeds_gen_mask'].to(accelerator.device)
                                     if batch['embeds_gen_mask'] is not None else None,
                                     embeds_cmp_mask=batch['embeds_cmp_mask'].to(accelerator.device)
                                     if batch['embeds_cmp_mask'] is not None else None,
                                     ids_gen_mask=batch['ids_gen_mask'].to(accelerator.device),
                                     ids_cmp_mask=batch['ids_cmp_mask'].to(accelerator.device),
                                     images_aug=batch['images_aug'].to(accelerator.device) if 'images_aug' in batch else None,
                )
                loss = output['total_loss']
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(agent_model.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
            metric = get_metric(output)
            print(metric)
            metric['lr'] = optimizer.param_groups[0]['lr']
            accelerator.log(metric, step=global_step)
            metric = {key: (format(value, ".6f") if isinstance(value, float) else value) for key, value in metric.items()}
            if accelerator.is_main_process:
                tqdm.write(str(metric))
            # # write the trained parameters name to file
            # if accelerator.is_main_process:
            #     with open(f'{args.output_dir}/trained_params.txt', 'w') as f:
            #         for name, param in agent_model.named_parameters():
            #             if param.requires_grad:
            #                 f.write(name + '\n')
            accelerator.wait_for_everyone()
            if global_step >= args.max_steps:
                break
            # if accelerator.is_main_process:
                # if global_step % args.eval_steps == 0:
                #     log_validation(diffusion_model_path, adapter, unet, vae, accelerator, weight_dtype, global_step, args.output_dir, args.validation_path)
                
            # TODO[huangzp]: fix validation bug in zero3, before that, use off-line evaluation strategy
            # if global_step % args.training_sample_steps == 0:
            #     should_log_sample = True
            # if 'recon_image_embeds' in output and should_log_sample and accelerator.is_main_process:
            #     torch.npu.empty_cache()
            #     recon_image_embeds = output['recon_image_embeds'].detach()
            #     target_embeds = output['target_embeds'].detach()
            #     mse_loss = output['mse_loss']
            #     # log_sample_training(pretrained_model_name_or_path, adapter, unet, vae, accelerator, weight_dtype, global_step=1, local_path=None, batch=None, recon_image_embeds, target_embeds):
            #     log_sample_training(recon_image_embeds, target_embeds, diffusion_model_path, adapter, unet, vae, accelerator, weight_dtype, global_step, mse_loss, args.output_dir)
            #     should_log_sample = False
            # accelerator.wait_for_everyone()
    accelerator.end_training()



from diffusers import StableDiffusionXLPipeline

# def log_sample_training(pretrained_model_name_or_path, adapter, unet, vae, accelerator, weight_dtype, global_step=1, local_path=None, batch=None):
@torch.no_grad()
def log_sample_training(recon_image_embeds, target_embeds, pretrained_model_name_or_path, adapter, unet, vae, accelerator, weight_dtype, global_step=1, mse_loss=0., local_path=None):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        tokenizer=None,
        tokenizer_2=None,
        text_encoder=None,
        text_encoder_2=None,
        torch_dtype=weight_dtype,
    )
    print("finished loading pipeline")

    pipeline = pipeline.to(accelerator.device)
    print("finished moving pipeline to device")
    pipeline.set_progress_bar_config(disable=True)
    # pipeline.

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    results = []

    image_prompt_embeds, pooled_image_prompt_embeds = adapter.encode_image_embeds(recon_image_embeds)
    # image_prompt_embeds = image_prompt_embeds.cpu()
    # pooled_image_prompt_embeds = pooled_image_prompt_embeds.cpu()
    # pipeline = pipeline.to('cpu')
    # generator = torch.Generator(device='cpu').manual_seed(42)
    
    
    # assert isinstance(image_aug, Image.Image)
    result = pipeline(
        prompt_embeds=image_prompt_embeds,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=pooled_image_prompt_embeds,
        negative_pooled_prompt_embeds=None,
        generator=generator,
        num_images_per_prompt=1,
        height=1024,
        width=1024,
    ).images[0]
    # save to local
    os.makedirs(f"{local_path}/training_sample_results", exist_ok=True)
    # add mse loss 0.0xxx to the tail of the path
    result.save(f"{local_path}/training_sample_results/train_sample_global_step_{global_step}_recon_image_embeds_output_mse_{mse_loss:.6f}.png")
    
    
    image_prompt_embeds, pooled_image_prompt_embeds = adapter.encode_image_embeds(target_embeds)
    
    
    # assert isinstance(image_aug, Image.Image)
    result = pipeline(
        prompt_embeds=image_prompt_embeds,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=pooled_image_prompt_embeds,
        negative_pooled_prompt_embeds=None,
        generator=generator,
        num_images_per_prompt=1,
        height=1024,
        width=1024,
    ).images[0]
    # save to local
    os.makedirs(f"{local_path}/training_sample_results", exist_ok=True)
    result.save(f"{local_path}/training_sample_results/train_sample_global_step_{global_step}_target_embeds_output.png")
    
    
    del pipeline
    torch.cuda.empty_cache()
    torch.npu.empty_cache()
    
    
def log_validation(pretrained_model_name_or_path, adapter, unet, vae, accelerator, weight_dtype, global_step, local_path, validation_path):
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        tokenizer=None,
        tokenizer_2=None,
        text_encoder=None,
        text_encoder_2=None,
        torch_dtype=weight_dtype,
    )
    print("finished loading pipeline")

    pipeline = pipeline.to(accelerator.device)
    print("finished moving pipeline to device")
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = torch.Generator(device=accelerator.device).manual_seed(42)
    results = []
    # with torch.cuda.amp.autocast():
    for val_image in os.listdir(f"{validation_path}/"):
        print(f"val_image: {val_image}")
        try:
            image = Image.open(f"{validation_path}/" + val_image)
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        assert isinstance(image, Image.Image)

        image_prompt_embeds, uncond_image_prompt_embeds, pooled_image_prompt_embeds, pooled_uncond_image_prompt_embeds = adapter.get_image_embeds(
            image_pil=image,
            return_negative=True,
            image_size=448,
        )
        
        images = pipeline(
            prompt_embeds=image_prompt_embeds,
            negative_prompt_embeds=uncond_image_prompt_embeds,
            pooled_prompt_embeds=pooled_image_prompt_embeds,
            negative_pooled_prompt_embeds=pooled_uncond_image_prompt_embeds,
            height=1024,
            width=1024,
            generator=generator,
        ).images
        results.extend(images)
        
    # save to local
    for i, image in enumerate(results):
        local_path = '.' if local_path is None else local_path
        os.makedirs(f"{local_path}/results", exist_ok=True)
        image.save(f"{local_path}/results/validation_sample_global_step_{global_step}_recon_images_{i}.png")

    # # wandb.log({"validation": [wandb.Image(image) for image in images]})
    # TODO[huangzp]: bug here, inference clean up schedule fix later
    # for tracker in accelerator.trackers:
    #     if tracker.name == "tensorboard":
    #         tensor_images = torch.cat([torchvision.transforms.ToTensor()(img).unsqueeze(0) for img in results], dim=0)
    #         # tensor_image = torchvision.utils.make_grid(tensor_image)
    #         tracker.writer.add_images(f"validation/global_step_{global_step}", tensor_images, global_step=global_step)

    del pipeline
    torch.cuda.empty_cache()
    torch.npu.empty_cache()



if __name__ == '__main__':
    train()
