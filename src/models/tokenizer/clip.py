from transformers import CLIPVisionModel


class CLIPVisionModelOneOutput(CLIPVisionModel):

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs).last_hidden_state