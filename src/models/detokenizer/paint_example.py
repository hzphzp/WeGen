import torch
import torch.nn as nn
from functools import partial
from einops import rearrange, repeat
from transformers import CLIPTokenizer, CLIPTextModel,CLIPVisionModel,CLIPModel
import kornia
# from ldm.modules.x_transformer import Encoder, TransformerWrapper  # TODO: can we directly rely on lucidrains code and simply add this as a reuirement? --> test
import math


import math

import torch as th
import torch.nn as nn


def convert_module_to_f16(l):
    """
    Convert primitive modules to float16.
    """
    if isinstance(l, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        l.weight.data = l.weight.data.half()
        if l.bias is not None:
            l.bias.data = l.bias.data.half()


class LayerNorm(nn.LayerNorm):
    """
    Implementation that supports fp16 inputs but fp32 gains/biases.
    """

    def forward(self, x: th.Tensor):
        return super().forward(x)


class MultiheadAttention(nn.Module):
    def __init__(self, n_ctx, width, heads):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        self.c_qkv = nn.Linear(width, width * 3)
        self.c_proj = nn.Linear(width, width)
        self.attention = QKVMultiheadAttention(heads, n_ctx)

    def forward(self, x):
        x = self.c_qkv(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.c_fc = nn.Linear(width, width * 4)
        self.c_proj = nn.Linear(width * 4, width)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class QKVMultiheadAttention(nn.Module):
    def __init__(self, n_heads: int, n_ctx: int):
        super().__init__()
        self.n_heads = n_heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.n_heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.n_heads, -1)
        q, k, v = th.split(qkv, attn_ch, dim=-1)
        weight = th.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = th.softmax(weight.float(), dim=-1).type(wdtype)
        return th.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        heads: int,
    ):
        super().__init__()

        self.attn = MultiheadAttention(
            n_ctx,
            width,
            heads,
        )
        self.ln_1 = LayerNorm(width)
        self.mlp = MLP(width)
        self.ln_2 = LayerNorm(width)

    def forward(self, x: th.Tensor):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        n_ctx: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx,
                    width,
                    heads,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: th.Tensor):
        for block in self.resblocks:
            x = block(x)
        return x

# GPUS_PER_NODE = torch.cuda.device_count()
# def dev():
#     """
#     Get the device to use for torch.distributed.
#     """
#     if torch.cuda.is_available():
#         return torch.device(f"cuda:{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}")
#     return torch.device("cpu")

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError

class QKVAttention_encoder(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads, compress, origin_num_tokens=256):
        super().__init__()
        self.n_heads = n_heads
        self.compress=compress
        self.proj_q=nn.Linear(origin_num_tokens, self.compress)

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        # assert width % (3 * self.n_heads) == 0
        ch = width // (3)
        # ch = (width-self.compress) // (2 * self.n_heads)
        # q = qkv[:,:self.compress,:].reshape(bs * self.n_heads, -1, length)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        q = self.proj_q(q)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, self.compress)


class ClassEmbedder(nn.Module):
    def __init__(self, embed_dim, n_classes=1000, key='class'):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)

    def forward(self, batch, key=None):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        c = self.embedding(c)
        return c


# class TransformerEmbedder(AbstractEncoder):
#     """Some transformer encoder layers"""
#     def __init__(self, n_embed, n_layer, vocab_size, max_seq_len=77, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
#                                               attn_layers=Encoder(dim=n_embed, depth=n_layer))

#     def forward(self, tokens):
#         tokens = tokens.to(self.device)  # meh
#         z = self.transformer(tokens, return_embeddings=True)
#         return z

#     def encode(self, x):
#         return self(x)


class BERTTokenizer(AbstractEncoder):
    """ Uses a pretrained BERT tokenizer by huggingface. Vocab size: 30522 (?)"""
    def __init__(self, device="cuda", vq_interface=True, max_length=77):
        super().__init__()
        from transformers import BertTokenizerFast  # TODO: add to reuquirements
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.device = device
        self.vq_interface = vq_interface
        self.max_length = max_length

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        return tokens

    @torch.no_grad()
    def encode(self, text):
        tokens = self(text)
        if not self.vq_interface:
            return tokens
        return None, None, [None, None, tokens]

    def decode(self, text):
        return text


# class BERTEmbedder(AbstractEncoder):
#     """Uses the BERT tokenizr model and add some transformer encoder layers"""
#     def __init__(self, n_embed, n_layer, vocab_size=30522, max_seq_len=77,
#                  device="cuda",use_tokenizer=True, embedding_dropout=0.0):
#         super().__init__()
#         self.use_tknz_fn = use_tokenizer
#         if self.use_tknz_fn:
#             self.tknz_fn = BERTTokenizer(vq_interface=False, max_length=max_seq_len)
#         self.device = device
#         self.transformer = TransformerWrapper(num_tokens=vocab_size, max_seq_len=max_seq_len,
#                                               attn_layers=Encoder(dim=n_embed, depth=n_layer),
#                                               emb_dropout=embedding_dropout)

#     def forward(self, text):
#         if self.use_tknz_fn:
#             tokens = self.tknz_fn(text)#.to(self.device)
#         else:
#             tokens = text
#         z = self.transformer(tokens, return_embeddings=True)
#         return z

#     def encode(self, text):
#         # output of length 77
#         return self(text)


class SpatialRescaler(nn.Module):
    def __init__(self,
                 n_stages=1,
                 method='bilinear',
                 multiplier=0.5,
                 in_channels=3,
                 out_channels=None,
                 bias=False):
        super().__init__()
        self.n_stages = n_stages
        assert self.n_stages >= 0
        assert method in ['nearest','linear','bilinear','trilinear','bicubic','area']
        self.multiplier = multiplier
        self.interpolator = partial(torch.nn.functional.interpolate, mode=method)
        self.remap_output = out_channels is not None
        if self.remap_output:
            print(f'Spatial Rescaler mapping from {in_channels} to {out_channels} channels after resizing.')
            self.channel_mapper = nn.Conv2d(in_channels,out_channels,1,bias=bias)

    def forward(self,x):
        for stage in range(self.n_stages):
            x = self.interpolator(x, scale_factor=self.multiplier)


        if self.remap_output:
            x = self.channel_mapper(x)
        return x

    def encode(self, x):
        return self(x)

# class FrozenCLIPEmbedder(AbstractEncoder):
#     """Uses the CLIP transformer encoder for text (from Hugging Face)"""
#     def __init__(self, version="openai/clip-vit-large-patch14", device=dev(), max_length=77):
#         super().__init__()
#         self.tokenizer = CLIPTokenizer.from_pretrained(version)
#         self.transformer = CLIPTextModel.from_pretrained(version)
#         self.device = device
#         self.max_length = max_length
#         self.freeze()

#     def freeze(self):
#         self.transformer = self.transformer.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
#                                         return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
#         tokens = batch_encoding["input_ids"].to(self.transformer.device)
#         outputs = self.transformer(input_ids=tokens)

#         z = outputs.last_hidden_state
#         return z

#     def encode(self, text):
#         return self(text)



class FrozenCLIPImageEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14"):
        super().__init__()
        self.transformer = CLIPVisionModel.from_pretrained(version)
        self.final_ln = LayerNorm(1024)
        self.mapper = Transformer(
                1,
                1024,
                5,
                1,
            )

        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False
        for param in self.mapper.parameters():
            param.requires_grad = True
        for param in self.final_ln.parameters():
            param.requires_grad = True

    def forward(self, image):
        outputs = self.transformer(pixel_values=image)
        z = outputs.pooler_output
        z = z.unsqueeze(1)
        z = self.mapper(z)
        z = self.final_ln(z)
        return z

    def encode(self, image,inpaint):
        return self(image)


# class FrozenCLIPTextEmbedder(nn.Module):
#     """
#     Uses the CLIP transformer encoder for text.
#     """
#     def __init__(self, version='ViT-L/14', device="cuda", max_length=77, n_repeat=1, normalize=True):
#         super().__init__()
#         self.model, _ = clip.load(version, jit=False, device="cpu")
#         self.device = device
#         self.max_length = max_length
#         self.n_repeat = n_repeat
#         self.normalize = normalize

#     def freeze(self):
#         self.model = self.model.eval()
#         for param in self.parameters():
#             param.requires_grad = False

#     def forward(self, text):
#         tokens = clip.tokenize(text).to(self.device)
#         z = self.model.encode_text(tokens)
#         if self.normalize:
#             z = z / torch.linalg.norm(z, dim=1, keepdim=True)
#         return z

#     def encode(self, text):
#         z = self(text)
#         if z.ndim==2:
#             z = z[:, None, :]
#         z = repeat(z, 'b 1 d -> b k d', k=self.n_repeat)
#         return z


# class FrozenClipImageEmbedder(nn.Module):
#     """
#         Uses the CLIP image encoder.
#         """
#     def __init__(
#             self,
#             model,
#             jit=False,
#             device='cuda' if torch.cuda.is_available() else 'cpu',
#             antialias=False,
#         ):
#         super().__init__()
#         self.model, _ = clip.load(name=model, device=device, jit=jit)

#         self.antialias = antialias

#         self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]), persistent=False)
#         self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]), persistent=False)

#     def preprocess(self, x):
#         # normalize to [0,1]
#         x = kornia.geometry.resize(x, (224, 224),
#                                    interpolation='bicubic',align_corners=True,
#                                    antialias=self.antialias)
#         x = (x + 1.) / 2.
#         # renormalize according to clip
#         x = kornia.enhance.normalize(x, self.mean, self.std)
#         return x

#     def forward(self, x):
#         # x is assumed to be in range [-1,1]
#         return self.model.encode_image(self.preprocess(x))

from transformers.modeling_outputs import BaseModelOutputWithPooling

class FrozenCLIPImageEmbedder_new(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, 
                 compress=2, 
                 input_dim=1024,
                 input_num_tokens=256,
                 compress_dim=2048,
                 compress_pooled_dim=1280,
                 ):
        super().__init__()
        self.compress = compress
        self.input_dim = input_dim
        self.input_num_tokens = input_num_tokens
        # self.global_img_feat = global_img_feat
        # self.concat_global_feat = concat_global_feat
        # self.official_projection=official_projection
        
        self.final_ln = LayerNorm(input_dim)
        self.transformer_img_pre = Transformer(
                1,
                input_dim,
                3,
                1,
            )
        
        self.encoder_qkv=nn.Linear(input_dim, input_dim*3)
        self.encoder_attention=QKVAttention_encoder(1, self.compress, origin_num_tokens=input_num_tokens)
        
        self.transformer_img_post = Transformer(
                1,
                input_dim,
                3,
                1,
            )
        
        self.output_proj = nn.Linear(input_dim, compress_dim)
        self.output_proj_pool = nn.Linear(input_dim, compress_pooled_dim)

    def forward(self, outputs):
        # outputs = self.transformer(pixel_values=image)
        # print(outputs.last_hidden_state.shape)
        # print(outputs.pooler_output.shape)
        # if self.global_img_feat:
        #     z = outputs.pooler_output
        #     z = z.unsqueeze(1)
        #     # print(z.shape)
        # else:
        if type(outputs) is BaseModelOutputWithPooling:
            z = outputs.last_hidden_state
        else:
            z = outputs
        z = self.transformer_img_pre(z)
        z = self.final_ln(z)
        # print(z.shape)
        z_temp=z[:,:self.input_num_tokens]
        z_temp=self.encoder_qkv(z_temp)
        z_temp=self.encoder_attention(z_temp.permute(0, 2, 1))
        z = z_temp.permute(0, 2, 1)
        
        z = self.transformer_img_post(z)

        # if not self.concat_global_feat:

        # else:
        #     z = torch.cat([z[:,0].unsqueeze(1),z_temp.permute(0, 2, 1)],dim=1)
        
        # return additional text embedding
        # pooled_z = z.mean(dim=1)
        z_out = self.output_proj(z)
        z_out_pool = self.output_proj_pool(torch.mean(z, dim=1))
        return z_out, z_out_pool

    def encode(self, image):
        return self(image)
    


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


if __name__ == "__main__":
    model = FrozenCLIPImageEmbedder_new()
    count_params(model, verbose=True)