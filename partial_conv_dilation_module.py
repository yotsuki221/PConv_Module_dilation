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

        # パディング幅 (指定がなければ 0)
        self.pad = padding if padding is not None else 0

        # Partial Convolution
        self.pconv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            padding=self.pad,
            dilation=self.dilation,
            bias=bias
        )

        # 重み付けマスク用畳み込み (非学習、重みは1)
        self.mask_conv = nn.Conv2d(
            1, 1,
            kernel_size=self.kernel_size,
            padding=self.pad,
            dilation=self.dilation,
            bias=False
        )

        # 重みを1に固定
        with torch.no_grad():
            self.mask_conv.weight.data.fill_(1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x) :
        # 入力サイズ取得
        n, c, h, w = x.shape

        # 入力を畳み込み
        out = self.pconv(x)

        # マスク(全1画像)
        mask = torch.ones(1, 1, h, w, device=x.device, dtype=x.dtype)

        # 有効画素数の算出
        mask_sum = self.mask_conv(mask)

        # カーネル内の要素数
        one_sum = float(self.kernel_size * self.kernel_size)

        out = out * (one_sum / mask_sum)

        return out