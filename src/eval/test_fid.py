from tqdm import tqdm 
import numpy as np 
import argparse 
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
    device = 'npu'
except:
    print('use NVIDIA GPU')
    device = 'cuda'
from PIL import Image
import json
import glob 
import time 
import os 
from src.eval.coco_eval.coco_evaluator import evaluate_model, compute_clip_score

device = 'cpu'
torch.set_grad_enabled(False)

parser = argparse.ArgumentParser() 
parser.add_argument("--seed", type=int, default=10)
parser.add_argument("--eval_res", type=int, default=256)
parser.add_argument("--ref_dir", type=str, default="path_to_coco_val")
parser.add_argument("--result_path", type=str, required=True, help="Path to the directory containing images to evaluate")

args = parser.parse_args()


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True 


print(f"eval {args.result_path}")
files = os.listdir(args.result_path)

# 读取图片文件并转换为Tensor
images = []
for i, file in enumerate(files):
    if file.endswith(('.png', '.jpg', '.jpeg')):  # 只处理png, jpg, jpeg文件
        image = Image.open(os.path.join(args.result_path, file))
        image = np.array(image)
        images.append(image)
    if (i + 1) % 1000 == 0:
        print(i + 1, " OK!", flush=True)
all_images = np.stack(images, axis=0)
print("numpy over", flush=True)

fid = evaluate_model(
    args, device, all_images, patch_fid=False
)

print(f"fid {fid}")