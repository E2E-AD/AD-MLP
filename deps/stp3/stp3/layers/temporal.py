from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from stp3.layers.convolutions import Bottleneck, Block, Bottleblock
from stp3.utils.geometry import warp_features


class SpatialGRU(nn.Module):
    """A GRU cell that takes an input tensor [BxTxCxHxW] and an optional previous state and passes a
    convolutional gated recurrent unit over the data"""

    def __init__(self, input_size, hidden_size, gru_bias_init=0.0):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder = nn.Conv2d(hidden_size, input_size, kernel_size=1, bias=False)

    def forward(self, x, state=None):
        # pylint: disable=unused-argument, arguments-differ
        # Check size
        assert len(x.size()) == 5, 'Input tensor must be BxTxCxHxW.'

        # recurrent layers
        rnn_output = []
        b, timesteps, c, h, w = x.size()
        rnn_state = torch.zeros(b, self.hidden_size, h, w, device=x.device) if state is None else state
        for t in range(timesteps):
            x_t = x[:, t]

            rnn_state = self.gru_cell(x_t, rnn_state)
            rnn_output.append(self.conv_decoder(rnn_state))

        # reshape rnn output to batch tensor
        return torch.stack(rnn_output, dim=1)

    def gru_cell(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update(x_and_state)
        reset_gate = self.conv_reset(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

class Dual_GRU(nn.Module):
    def __init__(self, in_channels, latent_dim, n_future, mixture=True, gru_bias_init = 0.0):
        super(Dual_GRU, self).__init__()

        input_size = in_channels
        hidden_size = latent_dim

        self.n_future = n_future
        self.mixture = mixture

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru_bias_init = gru_bias_init

        self.conv_update_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.conv_update_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_2 = nn.Conv2d(hidden_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)

        self.trusting_gate = nn.Sequential(
            Bottleblock(hidden_size+hidden_size, hidden_size),
            nn.Conv2d(hidden_size, 2, kernel_size=1, bias=False)
        )

    def forward(self, x, state):
        '''
        x: torch.Tensor [b, 1, input_size, h, w]
        state: torch.Tensor [b, n_present, hidden_size, h, w]
        '''
        b, s, c, h, w = x.shape
        assert c == self.input_size, f'feature sizes must match, got input {c} for layer with size {self.input_size}'
        n_present = state.shape[1]

        h = state[:, 0]
        pred_state = []

        # warmup
        for t in range(n_present - 1):
            cur_state = state[:, t]
            h = self.gru_cell_2(cur_state, h)

        # recurrent layers
        rnn_state1 = state[:, -1]
        rnn_state2 = state[:, -1]
        x = x[:, 0]

        for t in range(self.n_future):
            # propagate gru v1
            rnn_state1 = self.gru_cell_1(x, rnn_state1)
            # propagate gru v2
            h = self.gru_cell_2(rnn_state2, h)
            rnn_state2 = self.conv_decoder_2(h)

            # mix the two distribution
            mix_state = torch.cat([rnn_state1, rnn_state2], dim=1)
            trust_gate = self.trusting_gate(mix_state)
            trust_gate = torch.softmax(trust_gate, dim=1)
            cur_state = rnn_state2 * trust_gate[:,0:1] + rnn_state1 * trust_gate[:,1:]

            pred_state.append(cur_state)

            if self.mixture:
                rnn_state1 = cur_state
                rnn_state2 = cur_state

        x = torch.stack(pred_state, dim=1)

        return x

    def gru_cell_1(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_1(x_and_state)
        reset_gate = self.conv_reset_1(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_1(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

    def gru_cell_2(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_2(x_and_state)
        reset_gate = self.conv_reset_2(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_2(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

class BiGRU(nn.Module):
    def __init__(self, in_channels, gru_bias_init = 0.0):
        super(BiGRU, self).__init__()
        input_size = in_channels
        hidden_size = in_channels

        self.input_size = in_channels
        self.hidden_size = in_channels
        self.gru_bias_init = gru_bias_init

        self.conv_update_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_1 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_1 = Bottleblock(hidden_size, input_size)

        self.conv_update_2 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_reset_2 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_state_tilde_2 = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size=3, bias=True, padding=1)
        self.conv_decoder_2 = Bottleblock(hidden_size, input_size)

        self.res_blocks = nn.Sequential(
            Bottleblock(in_channels + in_channels, in_channels),
            Block(in_channels, in_channels),
            Block(in_channels, in_channels)
        )


    def forward(self, x):
        '''
        x: torch.Tensor [b, s, input_size, h, w]
        '''
        b, s, c, h, w = x.shape

        rnn_state1 = x[:, 0]
        rnn_state2 = x[:, -1]

        states_1 = []
        states_2 = []

        for t in range(s):
            # propagate gru v1
            rnn_state1 = self.gru_cell_1(x[:, t], rnn_state1)
            # propagate gru v2
            rnn_state2 = self.gru_cell_2(x[:, s-t-1], rnn_state2)

            states_1.append(self.conv_decoder_1(rnn_state1))
            states_2.append(self.conv_decoder_2(rnn_state2))

        states_1 = torch.stack(states_1, dim=1)
        states_2 = torch.stack(states_2[::-1], dim=1)

        states = torch.cat([states_1, states_2], dim=2)

        b, s, c, h, w = states.shape
        x = self.res_blocks(states.view(b * s, c, h, w))
        x = x.view(b, s, *x.shape[1:])
        # reshape rnn output to batch tensor
        return x

    def gru_cell_1(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_1(x_and_state)
        reset_gate = self.conv_reset_1(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_1(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output

    def gru_cell_2(self, x, state):
        # Compute gates
        x_and_state = torch.cat([x, state], dim=1)
        update_gate = self.conv_update_2(x_and_state)
        reset_gate = self.conv_reset_2(x_and_state)
        # Add bias to initialise gate as close to identity function
        update_gate = torch.sigmoid(update_gate + self.gru_bias_init)
        reset_gate = torch.sigmoid(reset_gate + self.gru_bias_init)

        # Compute proposal state, activation is defined in norm_act_config (can be tanh, ReLU etc)
        state_tilde = self.conv_state_tilde_2(torch.cat([x, (1.0 - reset_gate) * state], dim=1))

        output = (1.0 - update_gate) * state + update_gate * state_tilde
        return output


class CausalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = (kernel_size[0] - 1) * dilation[0]
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        # Pad temporally on the left
        # padding (padding_left, padding_right, padding_top, padding_bottom, padding_front, padding_back)
        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        (x,) = inputs
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class TemporalConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, n_present, n_future, kernel_size=(2, 3, 3), dilation=(1, 1, 1), bias=False):
        super(TemporalConv3d, self).__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'

        time_pad = n_future - n_present + dilation[0] * (kernel_size[0] - 1)
        height_pad = ((kernel_size[1] - 1) * dilation[1]) // 2
        width_pad = ((kernel_size[2] - 1) * dilation[2]) // 2

        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad // 2, time_pad - time_pad // 2), value=0)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, dilation=dilation, stride=1, padding=0, bias=bias)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class CausalMaxPool3d(nn.Module):
    def __init__(self, kernel_size=(2, 3, 3)):
        super().__init__()
        assert len(kernel_size) == 3, 'kernel_size must be a 3-tuple.'
        time_pad = kernel_size[0] - 1
        height_pad = (kernel_size[1] - 1) // 2
        width_pad = (kernel_size[2] - 1) // 2

        # Pad temporally on the left
        self.pad = nn.ConstantPad3d(padding=(width_pad, width_pad, height_pad, height_pad, time_pad, 0), value=0)
        self.max_pool = nn.MaxPool3d(kernel_size, stride=1)

    def forward(self, *inputs):
        (x,) = inputs
        x = self.pad(x)
        x = self.max_pool(x)
        return x


def conv_1x1x1_norm_activated(in_channels, out_channels):
    """1x1x1 3D convolution, normalization and activation layer."""
    return nn.Sequential(
        OrderedDict(
            [
                ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)),
                ('norm', nn.BatchNorm3d(out_channels)),
                ('activation', nn.ReLU(inplace=True)),
            ]
        )
    )


class Bottleneck3D(nn.Module):
    """
    Defines a bottleneck module with a residual connection
    """

    def __init__(self, in_channels, out_channels=None, kernel_size=(2, 3, 3), dilation=(1, 1, 1)):
        super().__init__()
        bottleneck_channels = in_channels // 2
        out_channels = out_channels or in_channels

        self.layers = nn.Sequential(
            OrderedDict(
                [
                    # First projection with 1x1 kernel
                    ('conv_down_project', conv_1x1x1_norm_activated(in_channels, bottleneck_channels)),
                    # Second conv block
                    (
                        'conv',
                        CausalConv3d(
                            bottleneck_channels,
                            bottleneck_channels,
                            kernel_size=kernel_size,
                            dilation=dilation,
                            bias=False,
                        ),
                    ),
                    # Final projection with 1x1 kernel
                    ('conv_up_project', conv_1x1x1_norm_activated(bottleneck_channels, out_channels)),
                ]
            )
        )

        if out_channels != in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.projection = None

    def forward(self, *args):
        (x,) = args
        x_residual = self.layers(x)
        x_features = self.projection(x) if self.projection is not None else x
        return x_residual + x_features


class PyramidSpatioTemporalPooling(nn.Module):
    """ Spatio-temporal pyramid pooling.
        Performs 3D average pooling followed by 1x1x1 convolution to reduce the number of channels and upsampling.
        Setting contains a list of kernel_size: usually it is [(2, h, w), (2, h//2, w//2), (2, h//4, w//4)]
    """

    def __init__(self, in_channels, reduction_channels, pool_sizes):
        super().__init__()
        self.features = []
        for pool_size in pool_sizes:
            assert pool_size[0] == 2, (
                "Time kernel should be 2 as PyTorch raises an error when" "padding with more than half the kernel size"
            )
            stride = (1, *pool_size[1:])
            padding = (pool_size[0] - 1, 0, 0)
            self.features.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            # Pad the input tensor but do not take into account zero padding into the average.
                            (
                                'avgpool',
                                torch.nn.AvgPool3d(
                                    kernel_size=pool_size, stride=stride, padding=padding, count_include_pad=False
                                ),
                            ),
                            ('conv_bn_relu', conv_1x1x1_norm_activated(in_channels, reduction_channels)),
                        ]
                    )
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, *inputs):
        (x,) = inputs
        b, _, t, h, w = x.shape
        # Do not include current tensor when concatenating
        out = []
        for f in self.features:
            # Remove unnecessary padded values (time dimension) on the right
            x_pool = f(x)[:, :, :-1].contiguous()
            c = x_pool.shape[1]
            x_pool = nn.functional.interpolate(
                x_pool.view(b * t, c, *x_pool.shape[-2:]), (h, w), mode='bilinear', align_corners=False
            )
            x_pool = x_pool.view(b, c, t, h, w)
            out.append(x_pool)
        out = torch.cat(out, 1)
        return out


class TemporalBlock(nn.Module):
    """ Temporal block with the following layers:
        - 2x3x3, 1x3x3, spatio-temporal pyramid pooling
        - dropout
        - skip connection.
    """

    def __init__(self, in_channels, out_channels=None, use_pyramid_pooling=False, pool_sizes=None):
        super().__init__()
        self.in_channels = in_channels
        self.half_channels = in_channels // 2
        self.out_channels = out_channels or self.in_channels
        self.kernels = [(2, 3, 3), (1, 3, 3)]

        # Flag for spatio-temporal pyramid pooling
        self.use_pyramid_pooling = use_pyramid_pooling

        # 3 convolution paths: 2x3x3, 1x3x3, 1x1x1
        self.convolution_paths = []
        for kernel_size in self.kernels:
            self.convolution_paths.append(
                nn.Sequential(
                    conv_1x1x1_norm_activated(self.in_channels, self.half_channels),
                    CausalConv3d(self.half_channels, self.half_channels, kernel_size=kernel_size),
                )
            )
        self.convolution_paths.append(conv_1x1x1_norm_activated(self.in_channels, self.half_channels))
        self.convolution_paths = nn.ModuleList(self.convolution_paths)

        agg_in_channels = len(self.convolution_paths) * self.half_channels

        if self.use_pyramid_pooling:
            assert pool_sizes is not None, "setting must contain the list of kernel_size, but is None."
            reduction_channels = self.in_channels // 3
            self.pyramid_pooling = PyramidSpatioTemporalPooling(self.in_channels, reduction_channels, pool_sizes)
            agg_in_channels += len(pool_sizes) * reduction_channels

        # Feature aggregation
        self.aggregation = nn.Sequential(
            conv_1x1x1_norm_activated(agg_in_channels, self.out_channels),)

        if self.out_channels != self.in_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(self.out_channels),
            )
        else:
            self.projection = None

    def forward(self, *inputs):
        (x,) = inputs
        x_paths = []
        for conv in self.convolution_paths:
            x_paths.append(conv(x))
        x_residual = torch.cat(x_paths, dim=1)
        if self.use_pyramid_pooling:
            x_pool = self.pyramid_pooling(x)
            x_residual = torch.cat([x_residual, x_pool], dim=1)
        x_residual = self.aggregation(x_residual)

        if self.out_channels != self.in_channels:
            x = self.projection(x)
        x = x + x_residual
        return x
