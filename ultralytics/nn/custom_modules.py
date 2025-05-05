import torch
import torch.nn as nn
from ultralytics.nn.modules import (
    Conv
)
import torch.nn.functional as F


class Add(nn.Module):
    """Adds tensors along a specified dimension for efficient tensor manipulation in neural networks."""

    def __init__(self):
        """Initializes an Add module to sum tensors element-wise."""
        super().__init__()

    def forward(self, x):
        """Adds a list of tensors element-wise; `x` is a list of tensors."""
        return sum(x)

class CA(nn.Module):
    def __init__(self, in_channels, reduction=1):
        super(CA, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        
        # Global Average Pooling
        avg_out = self.global_avg_pool(x).view(batch_size, channels)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)
        avg_out = self.sigmoid(avg_out).view(batch_size, channels, 1, 1)

        # Scale the original feature map
        return x * avg_out
    
class ConvCA(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, reduction=1):
        super().__init__()
        self.conv = Conv(c1, c2, k, s, p, g, d, act)
        self.ca = CA(c2, reduction)

    def forward(self, x):
        x = self.conv(x)
        x = self.ca(x)
        return x

    

class MBH(nn.Module):
    def __init__(self, input_channels=3, intermediate_channels=64, output_channels=64):
        super(MBH, self).__init__()
        # Branch 1: Depthwise separable convolution to focus on spatial-aware features
        self.branch1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, kernel_size=3, padding=1, groups=input_channels, bias=False),  # Depthwise
            nn.BatchNorm2d(input_channels),
            nn.ReLU()
        )
        # Branch 2: 3x3 convolution to increase receptive field and expand spectral channel dimension
        self.branch2 = nn.Sequential(
            nn.Conv2d(input_channels, intermediate_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(intermediate_channels),
            nn.ReLU()
        )

        # Fully connected layers for channel reduction and restoration
        reduced_channels = (input_channels + intermediate_channels) // 8
        self.fc_avg = nn.Sequential(
            nn.Linear(input_channels + intermediate_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, input_channels + intermediate_channels)
        )
        self.fc_max = nn.Sequential(
            nn.Linear(input_channels + intermediate_channels, reduced_channels),
            nn.ReLU(),
            nn.Linear(reduced_channels, input_channels + intermediate_channels)
        )

        # Final 1x1 convolution to reduce channels and adjust feature weights
        self.final_conv = nn.Conv2d(input_channels + intermediate_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # Process through both branches
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)

        # Concatenate outputs from both branches
        combined = torch.cat([branch1_out, branch2_out], dim=1)  # Concatenation along the channel dimension

        # Dynamically determine kernel size for pooling
        batch_size, _, height, width = combined.size()
        kernel_size = (height, width)

        # Compute average pooling
        avg_pool_out = nn.functional.avg_pool2d(combined, kernel_size).view(batch_size, -1)  # Flatten for FC layers

        # Compute max pooling
        max_pool_out = nn.functional.max_pool2d(combined, kernel_size).view(batch_size, -1)  # Flatten for FC layers

        # Process through fully connected layers
        fc_avg_out = self.fc_avg(avg_pool_out)
        fc_max_out = self.fc_max(max_pool_out)

        # Sum the outputs of the fully connected layers
        channel_weights = torch.sigmoid(fc_avg_out + fc_max_out).view(batch_size, -1, 1, 1)  # Reshape for broadcasting

        # Apply channel weights to the combined features
        weighted_features = combined * channel_weights

        # Final convolution to reduce channels and adjust weights
        out = self.final_conv(weighted_features)
        return out
    
class TA(nn.Module):
    def __init__(self, channels):
        super(TA, self).__init__()
        self.channels = channels

        # Channel Attention Layers
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling for MeanPool(F)
        self.channel_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # Spatial Attention Layers
        self.spatial_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.para = nn.Parameter(torch.ones(1))  # Learnable parameter for spatial threshold

    def forward(self, x):
        # Channel Attention
        B, C, H, W = x.size()
        
        Fc = self.avg_pool(x)  # Fc: (B, C, 1, 1)
        Mbarc = self.channel_conv(Fc)  # Mbarc = Conv(Fc)
        Mc = self.sigmoid(Fc - Mbarc) * Fc  # Mc = Sigmoid(Fc - Mbarc) * Fc

        # spatial Attention
        conv_x = self.spatial_conv(x)  # Apply Conv to get spatial feature map
        Fs1 = F.max_pool2d(conv_x, kernel_size=(1, conv_x.shape[-1])).transpose(2, 3)  # along width dem   ex:-  (1, 1, 1, 80) after T
        Fs2 = F.max_pool2d(conv_x, kernel_size=(conv_x.shape[-2], 1))                  # along height dem   ex:-  (1, 1, 1, 80)
        Fs = torch.cat((Fs1, Fs2), dim=3)  # Shape will be [1, 1, 1, 160] after concatenation
        Mbars = self.para * torch.max(Fs, dim=3, keepdim=True)[0]  # Mbars = para * Max(Fs)
        spatial_attention = self.sigmoid(Fs - Mbars) * Fs
        Mh, Mw = torch.split(spatial_attention, [H, W], dim=3)  # make usable to height and dimensions are not same images

        Ms = Mw + Mh.transpose(2, 3)
        out = F.relu(self.channel_conv(Mc * Ms)) + x
        return out


########################### CARAFE old ##########################
class CARAFE(nn.Module):
    """CARAFE: Content-Aware ReAssembly of FEatures https://arxiv.org/pdf/1905.02188.pdf"""

    def __init__(self, c1, c2, kernel_size=3, up_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(c1, c1 // 4, 1)
        self.encoder = nn.Conv2d(c1 // 4, self.up_factor ** 2 * self.kernel_size ** 2, self.kernel_size, 1,
                                 self.kernel_size // 2)
        self.out = nn.Conv2d(c1, c2, 1)

    def forward(self, x):
        N, C, H, W = x.size()
        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(N, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        x = F.pad(x,
                  pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant',
                  value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        x = x.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        x = x.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        x = x.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        x = x.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(x, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor




################# CARAFE new #######################
# class ConvBNReLU(nn.Module):
#     '''Module for the Conv-BN-ReLU tuple.'''
#     def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
#                  use_relu=True):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(
#                 c_in, c_out, kernel_size=kernel_size, stride=stride, 
#                 padding=padding, dilation=dilation, bias=False)
#         self.bn = nn.BatchNorm2d(c_out)
#         if use_relu:
#             self.relu = nn.ReLU(inplace=True)
#         else:
#             self.relu = None

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x


# class CARAFE(nn.Module):
#     def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
#         """ The unofficial implementation of the CARAFE module.

#         The details are in "https://arxiv.org/abs/1905.02188".

#         Args:
#             c: The channel number of the input and the output.
#             c_mid: The channel number after compression.
#             scale: The expected upsample scale.
#             k_up: The size of the reassembly kernel.
#             k_enc: The kernel size of the encoder.

#         Returns:
#             X: The upsampled feature map.
#         """
#         super(CARAFE, self).__init__()
#         self.scale = scale

#         self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1, 
#                                padding=0, dilation=1)
#         self.enc = ConvBNReLU(c_mid, (scale*k_up)**2, kernel_size=k_enc, 
#                               stride=1, padding=k_enc//2, dilation=1, 
#                               use_relu=False)
#         self.pix_shf = nn.PixelShuffle(scale)

#         self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
#         self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale, 
#                                 padding=k_up//2*scale)

#     def forward(self, X):
#         b, c, h, w = X.size()
#         h_, w_ = h * self.scale, w * self.scale
        
#         W = self.comp(X)                                # b * m * h * w
#         W = self.enc(W)                                 # b * 100 * h * w
#         W = self.pix_shf(W)                             # b * 25 * h_ * w_
#         W = F.softmax(W, dim=1)                         # b * 25 * h_ * w_

#         X = self.upsmp(X)                               # b * c * h_ * w_
#         X = self.unfold(X)                              # b * 25c * h_ * w_
#         X = X.view(b, c, -1, h_, w_)                    # b * 25 * c * h_ * w_

#         X = torch.einsum('bkhw,bckhw->bchw', [W, X])    # b * c * h_ * w_
#         return X