import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .modules import *


####################################
# DEFINE MODEL INSTANCES
####################################


class RT4KSR_Rep(nn.Module):
    def __init__(self,
                 num_channels,
                 num_feats,
                 num_blocks,
                 upscale,
                 act,
                 eca_gamma,
                 is_train,
                 forget,
                 layernorm,
                 residual,
                 convlstm_channels,
                 convlstm_kernel_size) -> None:
        super().__init__()
        self.forget = forget
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gaussian = torchvision.transforms.GaussianBlur(kernel_size=5, sigma=1)

        self.down = nn.PixelUnshuffle(2)
        self.up = nn.PixelShuffle(2)

        self.head = nn.Sequential(nn.Conv2d(num_channels * (2**2), num_feats, 3, padding=1))

        # ConvLSTM layer
        self.convlstm = ConvLSTM(num_feats, convlstm_channels, convlstm_kernel_size)
        self.hidden_state = None  # Initialize to None for the first frame

        hfb = []
        if is_train:
            hfb.append(ResBlock(num_feats, ratio=2))
        else:
            hfb.append((RepResBlock(num_feats)))
        hfb.append(act)
        self.hfb = nn.Sequential(*hfb)

        body = []
        for i in range(num_blocks):
            if is_train:
                body.append(SimplifiedNAFBlock(in_c=num_feats, act=act, exp=2, layernorm=layernorm, residual=residual))
            else:
                body.append(SimplifiedRepNAFBlock(in_c=num_feats, act=act, exp=2, layernorm=layernorm, residual=residual))

        self.body = nn.Sequential(*body)

        tail = [LayerNorm2d(num_feats)]
        self.tail = nn.Sequential(*tail)

        self.conv_main_scale = nn.Sequential(
            nn.Conv2d(num_feats, 3 * ((upscale*2) ** 2), 3, padding=1),
            nn.PixelShuffle(upscale * 2)
        )

        self.conv_skip_scale = nn.Sequential(
            nn.Conv2d(num_channels, 3 * ((upscale) ** 2), 5, padding=2),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x, external_hidden_state=None):
        if external_hidden_state is not None:
            hidden_state = external_hidden_state
        else:
            hidden_state = self.hidden_state

        # unshuffle to save computation
        # print(x.size())
        x_unsh = self.down(x)

        shallow_feats_lr = self.head(x_unsh)

        NAF_out = shallow_feats_lr
        for NAFBlock in self.body:
            NAF_out = NAFBlock(NAF_out)
        NAF_out = shallow_feats_lr + NAF_out
        # NAF_out, new_hidden_state = self.convlstm(NAF_out, hidden_state)

        deep_feats = self.tail(NAF_out)
        deep_feats, new_hidden_state = self.convlstm(deep_feats, hidden_state)

        main_out = self.conv_main_scale(deep_feats)

        skip_out = self.conv_skip_scale(x)
        out = main_out + skip_out

        self.hidden_state = new_hidden_state if external_hidden_state is None else None
        return out, new_hidden_state

    def reset_hidden_state(self):
        self.hidden_state = None  # Resets the hidden state
####################################
# RETURN INITIALIZED MODEL INSTANCES
####################################

def rt4ksr_rep(config):
    act = activation(config.act_type)
    model = RT4KSR_Rep(num_channels=3,
                       num_feats=config.feature_channels,
                       num_blocks=config.num_blocks,
                       upscale=config.scale,
                       act=act,
                       eca_gamma=1,
                       forget=False,
                       is_train=config.is_train,
                       layernorm=True,
                       residual=False,
                       convlstm_channels=config.feature_channels,
                       convlstm_kernel_size=3)
    return model


def test1():
    # init sample config to test model
    class Config:
        def __init__(self):
            self.feature_channels = 24
            self.num_blocks = 6
            self.act_type = "gelu"
            self.is_train = False
            self.scale = 2
    config = Config()

    from torchsummary import summary

    model = rt4ksr_rep(config)
    summary(model, (3, 128, 128), device='cpu')
    out = model(torch.randn(1, 3, 128, 128))
    print("Output shape:",out.shape)

if __name__=='__main__':
    act = activation("gelu")
    model = RT4KSR_Rep(num_channels=3,
                       num_feats=32,
                       num_blocks=6,
                       upscale=2,
                       act=act,
                       eca_gamma=1,
                       forget=False,
                       is_train=False,
                       layernorm=True,
                       residual=False,
                       convlstm_channels=32,
                       convlstm_kernel_size=3)

    from torchsummary import summary
    summary(model, (3, 128, 128), device='cpu')
