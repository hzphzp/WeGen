from libs.eva_clip import create_model, create_model_and_transforms, get_tokenizer
from PIL import Image
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
except:
    print('use NVIDIA GPU')
    
from torch import nn
import math
    
    
def resize_position_embeddings(visual, image_size=224):
    # new image size for all
    old_image_size = visual.image_size
    visual.image_size = image_size
    
    # new image size for patch_embed
    visual.patch_embed.img_size = (visual.image_size, visual.image_size)
    visual.patch_embed.num_patches = (visual.patch_embed.img_size[1] // visual.patch_embed.patch_size[1]) \
                                        * (visual.patch_embed.img_size[0] // visual.patch_embed.patch_size[0])
    
    # new image size for pos_embed
    # visual.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
    class_pos_embed = visual.pos_embed[:, 0]
    patch_pos_embed = visual.pos_embed[:, 1:]
    num_positions = visual.pos_embed.shape[1] - 1
    dim = visual.pos_embed.shape[-1]
    h0 = visual.image_size // visual.patch_embed.patch_size[0]
    w0 = visual.image_size // visual.patch_embed.patch_size[1]
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    h0, w0 = h0 + 0.1, w0 + 0.1
    patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
    patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
    # resize patch_pos_embed
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed,
        scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
        mode="bicubic",
        align_corners=False,
    )
    assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    resized_position_embedding = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).squeeze(0)
    visual.pos_embed = nn.Parameter(resized_position_embedding)
    return visual
    
    
def create_eva_clip_e_plus(image_size=448):
    model_path = "./wegen_mllm_ckpt/pretrained/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt"
    model_name = "EVA02-CLIP-bigE-14-plus"
    model, _, preprocess = create_model_and_transforms(model_name, model_path, force_custom_clip=True)
    visual = model.visual
    visual = resize_position_embeddings(visual, image_size)
    return visual



from omegaconf import OmegaConf
import hydra
if __name__ == '__main__':
    model_path = "./wegen_mllm_ckpt/pretrained/EVA-CLIP/EVA02_CLIP_E_psz14_plus_s9B.pt"
    model_name = "EVA02-CLIP-bigE-14-plus"
    model, _, preprocess = create_model_and_transforms(model_name, model_path, force_custom_clip=True)
    model.visual = resize_position_embeddings(model.visual, 448)
    preprocess_config = OmegaConf.load("configs/processer/qwen_448_transform_crop.yaml")
    preprocess = hydra.utils.instantiate(preprocess_config)
    tokenizer = get_tokenizer(model_name)
    device = 'npu'
    model = model.to(device)
    
    image_path = 'validation/woman_beach.jpg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(["a diagram", "a dog", "a cat", "a woman", "sea and ocean"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        print(image_features.shape)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("Label probs:", text_probs)  # [0.0519, 0.0642, 0.0167, 0.8072, 0.0600]
    # [0.0100, 0.0538, 0.0227, 0.5631, 0.3504]
    # [0.0519, 0.0642, 0.0167, 0.8072, 0.0600]

    image_path = 'person9.jpg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(["a boy in white shirt", "a girl", "a cat", "a woman", "sky", "sea and ocean"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        print(image_features.shape)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("Label probs:", text_probs)  # prints: [4.4523e-06, 3.9470e-01, 1.5524e-03, 7.4975e-02, 5.2319e-01, 5.5741e-03]
    # [7.2751e-06, 1.2631e-01, 1.4535e-03, 3.6102e-02, 8.1366e-01, 2.2466e-02]
    # [4.4523e-06, 3.9470e-01, 1.5524e-03, 7.4975e-02, 5.2320e-01, 5.5742e-03]
    
    image_path = 'validation/face7.jpeg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text =  tokenizer(["obama", "a man", "cat", "dog"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        print(image_features.shape)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("Label probs:", text_probs)  # [0.3097, 0.5963, 0.0319, 0.0621]
    # [0.9376, 0.0105, 0.0153, 0.0366]
    # [0.3097, 0.5964, 0.0319, 0.0621]
    
    image_path = 'validation/building1.jpg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(["a building", "a tower", "a car", "a tree", "a person"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        print(image_features.shape)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("Label probs:", text_probs)  # [0.2364, 0.0767, 0.0537, 0.2911, 0.3422]
    # [0.5770, 0.0217, 0.0596, 0.0499, 0.2918]
    # [0.2364, 0.0767, 0.0537, 0.2911, 0.3422]
    
    image_path = "validation/eagle.jpg"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(["an eagle", "a cat", "a dog", "a building", "a mountain"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        print(image_features.shape)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("Label probs:", text_probs)  # [0.4107, 0.0145, 0.0557, 0.4591, 0.0600]
    # [0.2364, 0.0029, 0.0070, 0.6997, 0.0539]
    # [0.4107, 0.0145, 0.0557, 0.4591, 0.0600]
    
    image_path = "validation/electrical3.jpg"
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = tokenizer(["Electrical appliances", "a cat", "a dog", "a building", "a mountain"]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        print(image_features.shape)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    print("Label probs:", text_probs)  # 
    # [0.0021, 0.0038, 0.0091, 0.9145, 0.0704]
    # [0.0088, 0.0244, 0.0936, 0.7722, 0.1010]
    
    
    import pdb; pdb.set_trace()
    
