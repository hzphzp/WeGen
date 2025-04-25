import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    print('use Ascend NPU')
except:
    print('use NVIDIA GPU')
from libs.mar.models.diffloss import DiffLoss


class DiffLossMultiStep(DiffLoss):
    def __init__(self, diffusion_batch_mul, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diffusion_batch_mul = diffusion_batch_mul
    def forward(self, target, z, mask=None):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        # mask = mask.reshape(bsz*seq_len).repeat(diffusion_batch_mul)
        loss = super().forward(z=z, target=target, mask=None)
        return loss


if __name__ == "__main__":
    # # Diffusion Loss
    # diffloss = DiffLoss(
    #     target_channels=1280,  # output feature dimension
    #     z_channels=1280,  # input condition dimension
    #     width=1280,  # width of the mlp network
    #     depth=8,  # depth of the mlp network
    #     num_sampling_steps='100',  #
    #     grad_checkpointing=False
    # )
    # diffloss = diffloss.cuda()
    # diffusion_batch_mul = 4

    # input_embed = torch.randn(8, 64, 1280).cuda()
    # target_embed = torch.randn(8, 64, 1280).cuda()
    # # print(input_embed)

    # def forward_loss(z, target):
    #     bsz, seq_len, _ = target.shape
    #     target = target.reshape(bsz * seq_len, -1).repeat(diffusion_batch_mul, 1)
    #     z = z.reshape(bsz*seq_len, -1).repeat(diffusion_batch_mul, 1)
    #     # mask = mask.reshape(bsz*seq_len).repeat(diffusion_batch_mul)
    #     loss = diffloss(z=z, target=target, mask=None)
    #     return loss

    # print(forward_loss(input_embed, target_embed))  # mask is only to mask the loss
    # print(diffloss.sample(z=input_embed.reshape(-1, 1280), temperature=1.0, cfg=1.0))
    diffloss = DiffLossMultiStep(
        target_channels=1280,  # output feature dimension
        z_channels=1280,  # input condition dimension
        width=1280,  # width of the mlp network
        depth=8,  # depth of the mlp network
        num_sampling_steps='100',  #
        grad_checkpointing=False,
        diffusion_batch_mul = 4,
    )
    diffloss = diffloss.cuda()


    input_embed = torch.randn(8, 64, 1280).cuda()
    target_embed = torch.randn(8, 64, 1280).cuda()
    # print(input_embed)

    print(diffloss(z=input_embed, target=target_embed))  # mask is only to mask the loss
    print(diffloss.sample(z=input_embed.reshape(-1, 1280), temperature=1.0, cfg=1.0))