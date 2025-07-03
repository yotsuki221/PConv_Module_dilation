import torch
import torch.nn.functional as F
from torch import nn

class PartialConvModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int = None,
                 dilation: int = 1,
                 bias: bool = False):
        super().__init__()

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.use_bias   = bias
        self.out_channels = out_channels

        # パディング幅 (指定がなければ 0)
        self.pad = padding if padding is not None else 0
        
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

        # Partial Convolution
        self.pconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            padding=self.pad,
            dilation=self.dilation,
            bias=False
        )
        

        if self.dilation == 1:
            # paddingしてavgpool
            self.make_mask = nn.AvgPool2d(
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.pad,
                count_include_pad=True  # ゼロパディングも平均計算に含める
            )
        else:
            # 重み付けマスク用畳み込み (非学習、重みは1)
            mask_conv = nn.Conv2d(
                1, 1,
                kernel_size=self.kernel_size,
                padding=self.pad,
                dilation=self.dilation,
                bias=False
            )
            # 重みを1に固定
            with torch.no_grad():
                mask_conv.weight.data.fill_(1.0 / (kernel_size * kernel_size))
            mask_conv.weight.requires_grad_(False)
            self.make_mask = mask_conv

    def forward(self, x) :
        # 入力サイズ取得
        n, c, h, w = x.shape

        # 入力を畳み込み
        out = self.pconv(x)

        # マスク(全1画像)
        mask = torch.ones(1, 1, h, w, device=x.device, dtype=x.dtype)

        out = out / self.make_mask(mask)

        if self.use_bias:
            out = out + self.bias.view(1, self.out_channels, 1, 1)

        return out
