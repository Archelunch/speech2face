import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
from tqdm import tqdm
from numba import jit


@jit(nopython=True)
def fill_mask(mask, type, rgb_last=True):
    input_dim = mask.shape[0]
    kernel_mid_y = mask.shape[-2] // 2
    kernel_mid_x = mask.shape[-1] // 2
    if rgb_last:
        if type == 'A':
            for i in range(input_dim):
                mask[i, i + 1:, :, :] = 0.0
                mask[i, i, kernel_mid_y + 1:, :] = 0.0
                mask[i, i, kernel_mid_y, kernel_mid_x + 1:] = 0.0
        if type == 'B':
            for i in range(input_dim):
                mask[i, :i, :, :] = 0.0
                mask[i, i, :kernel_mid_y, :] = 0.0
                mask[i, i, kernel_mid_y, :kernel_mid_x] = 0.0
    else:
        if type == 'A':
            for i in range(input_dim):
                mask[i, :, kernel_mid_y + 1:, :] = 0.0
                # For the current and previous color channels, including the current color
                mask[i, :i + 1, kernel_mid_y, kernel_mid_x + 1:] = 0.0
                # For the latter color channels, not including the current color
                mask[i, i + 1:, kernel_mid_y, kernel_mid_x:] = 0.0

        elif type == 'B':
            for i in range(input_dim):
                mask[i, :, :kernel_mid_y, :] = 0.0
                # For the current and latter color channels, including the current color
                mask[i, i:, kernel_mid_y, :kernel_mid_x] = 0.0
                # For the previous color channels, not including the current color
                mask[i, :i, kernel_mid_y, :kernel_mid_x + 1] = 0.0
        else:
            raise TypeError('type should be either A or B')


@jit(nopython=True)
def fill_center_mask(mask):
    input_dim = mask.shape[0]
    kernel_mid_y = mask.shape[-2] // 2
    kernel_mid_x = mask.shape[-1] // 2
    for i in range(input_dim):
        mask[i, i, kernel_mid_y, kernel_mid_x] = 1.0


@jit(nopython=True)
def generate_masks(mask1, center_mask1, mask2, center_mask2, mask3, center_mask3, input_dim, latent_dim, type, rgb_last):
    for i in range(latent_dim):
        fill_mask(mask1[i * input_dim: (i + 1) * input_dim, ...],
                  type=type, rgb_last=rgb_last)
        fill_center_mask(center_mask1[i * input_dim: (i + 1) * input_dim, ...])
        fill_mask(mask3[:, i * input_dim: (i + 1) *
                        input_dim, ...], type=type, rgb_last=rgb_last)
        fill_center_mask(
            center_mask3[:, i * input_dim: (i + 1) * input_dim, ...])
        for j in range(latent_dim):
            fill_mask(mask2[i * input_dim: (i + 1) * input_dim, j * input_dim: (j + 1) * input_dim, ...],
                      type=type, rgb_last=rgb_last)
            fill_center_mask(
                center_mask2[i * input_dim: (i + 1) * input_dim, j * input_dim: (j + 1) * input_dim, ...])


class SequentialWithSampling(nn.Sequential):
    def sampling(self, z):
        for module in reversed(self._modules.values()):
            z = module.sampling(z)
        return z


class BasicBlock(nn.Module):
    # Input_dim should be 1(grey scale image) or 3(RGB image), or other dimension if use SpaceToDepth
    def init_conv_weight(self, weight):
        init.xavier_normal_(weight, 0.005)

    def init_conv_bias(self, weight, bias):
        fan_in, _ = init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(bias, -bound, bound)

    def __init__(self, config, shape, latent_dim, type, input_dim=3, kernel1=3, kernel2=3, kernel3=3, init_zero=False):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.padding1 = kernel1 // 2
        self.padding2 = kernel2 // 2
        self.padding3 = kernel3 // 2
        self.kernel1 = kernel1
        self.kernel2 = kernel2
        self.kernel3 = kernel3

        self.weight1 = nn.Parameter(
            torch.randn(input_dim * latent_dim, input_dim,
                        kernel1, kernel1) * 1e-5
        )
        self.bias1 = nn.Parameter(
            torch.zeros(input_dim * latent_dim)
        )

        if not init_zero:
            self.init_conv_weight(self.weight1)
            self.init_conv_bias(self.weight1, self.bias1)

        self.weight2 = nn.Parameter(
            torch.randn(input_dim * latent_dim, input_dim *
                        latent_dim, kernel2, kernel2) * 1e-5
        )
        self.bias2 = nn.Parameter(
            torch.zeros(input_dim * latent_dim)
        )

        if not init_zero:
            self.init_conv_weight(self.weight2)
            self.init_conv_bias(self.weight2, self.bias2)

        self.weight3 = nn.Parameter(
            torch.randn(input_dim, input_dim * latent_dim,
                        kernel3, kernel3) * 1e-5
        )

        self.bias3 = nn.Parameter(
            torch.zeros(input_dim)
        )
        if not init_zero:
            self.init_conv_weight(self.weight3)
            self.init_conv_bias(self.weight3, self.bias3)

        # Define masks
        # Mask out the element above diagonal
        self.type = type
        self.mask1 = np.ones(self.weight1.shape, dtype=np.float32)
        self.center_mask1 = np.zeros(self.weight1.shape, dtype=np.float32)
        self.mask2 = np.ones(self.weight2.shape, dtype=np.float32)
        self.center_mask2 = np.zeros(self.weight2.shape, dtype=np.float32)
        self.mask3 = np.ones(self.weight3.shape, dtype=np.float32)
        self.center_mask3 = np.zeros(self.weight3.shape, dtype=np.float32)

        generate_masks(self.mask1, self.center_mask1, self.mask2, self.center_mask2, self.mask3, self.center_mask3,
                       input_dim, latent_dim, type, config.model.rgb_last)

        self.mask1 = nn.Parameter(torch.from_numpy(
            self.mask1), requires_grad=False)
        self.center_mask1 = nn.Parameter(torch.from_numpy(
            self.center_mask1), requires_grad=False)
        self.mask2 = nn.Parameter(torch.from_numpy(
            self.mask2), requires_grad=False)
        self.center_mask2 = nn.Parameter(torch.from_numpy(
            self.center_mask2), requires_grad=False)
        self.mask3 = nn.Parameter(torch.from_numpy(
            self.mask3), requires_grad=False)
        self.center_mask3 = nn.Parameter(torch.from_numpy(
            self.center_mask3), requires_grad=False)

        self.non_linearity = F.elu
        self.non_linearity_derivative = elu_derivative

        self.t = nn.Parameter(torch.ones(1, *shape))
        self.shape = shape
        self.config = config

    def forward(self, x):
        log_det = x[1]
        x = x[0]
        masked_weight1 = self.weight1 * self.mask1
        masked_weight3 = self.weight3 * self.mask3

        # shape: B x latent_output . input_dim x img_size x img_size
        latent_output = F.conv2d(
            x, masked_weight1, bias=self.bias1, padding=self.padding1, stride=1)

        kernel_mid_y, kernel_mid_x = masked_weight1.shape[-2] // 2, masked_weight1.shape[-1] // 2
        diag1 = torch.diagonal(
            masked_weight1[..., kernel_mid_y, kernel_mid_x].view(
                self.latent_dim, self.input_dim, self.input_dim),
            dim1=-2, dim2=-1)  # shape: latent_dim x input_dim

        diag1 = self.non_linearity_derivative(latent_output). \
            view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2], x.shape[-1]) \
            * diag1[None, :, :, None, None]  # shape: B x latent_dim x input_dim x img_shape x img_shape

        latent_output = self.non_linearity(latent_output)

        # shape: latent_dim.input_dim x input_dim x kernel x kernel
        center1 = masked_weight1 * self.center_mask1
        # shape: input_dim x latent_dim.input_dim x kernel x kernel
        center3 = masked_weight3 * self.center_mask3

        # shape: 1 x latent_dim x input_dim x input_dim x kernel x kernel
        center1 = center1.view(self.latent_dim, self.input_dim, self.input_dim,
                               center1.shape[-2], center1.shape[-1]).unsqueeze(0)

        # shape: latent_dim x 1 x input_dim x input_dim x kernel x kernel
        center3 = center3.view(self.input_dim, self.latent_dim, self.input_dim, center3.shape[-2],
                               center3.shape[-1]).permute(1, 0, 2, 3, 4).unsqueeze(1)

        sign_prods = torch.sign(center1) * torch.sign(center3)
        # shape: latent_dim.input_dim x latent_dim.input_dim x kernel x kernel
        center2 = self.weight2 * self.center_mask2

        center2 = center2.view(self.latent_dim, self.input_dim, self.latent_dim, self.input_dim,
                               center2.shape[-2], center2.shape[-1])

        center2 = center2.permute(0, 2, 1, 3, 4, 5)
        center2 = sign_prods[..., self.kernel3 // 2, self.kernel1 //
                             2].unsqueeze(-1).unsqueeze(-1) * torch.abs(center2)
        center2 = center2.permute(
            0, 2, 1, 3, 4, 5).contiguous().view_as(self.weight2)

        masked_weight2 = (center2 * self.center_mask2 +
                          self.weight2 * (1. - self.center_mask2)) * self.mask2

        latent_output = F.conv2d(
            latent_output, masked_weight2, bias=self.bias2, padding=self.padding2, stride=1)

        kernel_mid_y, kernel_mid_x = masked_weight2.shape[-2] // 2, masked_weight2.shape[-1] // 2
        diag2 = masked_weight2[..., kernel_mid_y, kernel_mid_x].view(self.latent_dim, self.input_dim, self.latent_dim,
                                                                     self.input_dim)
        diag2 = torch.diagonal(diag2.permute(0, 2, 1, 3), dim1=-2,
                               dim2=-1)  # shape: latent_dim x latent_dim x input_dim
        # shape: 1 x latent_dim x latent_dim x input_dim x 1 x 1
        diag2 = diag2[None, :, :, :, None, None]

        diag2 = torch.sum(diag2 * diag1.unsqueeze(1),
                          dim=2)  # shape: B x latent_dim x input_dim x img_shape x img_shape

        latent_output_derivative = self.non_linearity_derivative(latent_output)
        latent_output = self.non_linearity(latent_output)

        latent_output = F.conv2d(
            latent_output, masked_weight3, bias=self.bias3, padding=self.padding3, stride=1)

        kernel_mid_y, kernel_mid_x = masked_weight3.shape[-2] // 2, masked_weight3.shape[-1] // 2
        diag3 = masked_weight3[..., kernel_mid_y, kernel_mid_x].view(
            self.input_dim, self.latent_dim, self.input_dim)
        # shape: latent_dim x input_dim
        diag3 = torch.diagonal(diag3.permute(1, 0, 2), dim1=-2, dim2=-1)
        diag3 = latent_output_derivative.view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2], x.shape[-1]) \
            * diag3[None, :, :, None, None]  # shape: B x latent_dim x input_dim x img_shape x img_shape

        # shape: B x input_dim x img_shape x img_shape
        diag = torch.sum(diag2 * diag3, dim=1)

        t = torch.max(torch.abs(self.t), torch.tensor(1e-12, device=x.device))
        output = latent_output + t * x
        log_det += torch.sum(torch.log(diag + t), dim=(1, 2, 3))

        return output, log_det

    def sampling(self, z):
        with torch.no_grad():
            masked_weight1 = self.weight1 * self.mask1
            masked_weight3 = self.weight3 * self.mask3
            # shape: latent_dim.input_dim x input_dim x kernel x kernel
            center1 = masked_weight1 * self.center_mask1
            # shape: input_dim x latent_dim.input_dim x kernel x kernel
            center3 = masked_weight3 * self.center_mask3

            # shape: 1 x latent_dim x input_dim x input_dim x kernel x kernel
            center1 = center1.view(self.latent_dim, self.input_dim, self.input_dim,
                                   center1.shape[-2], center1.shape[-1]).unsqueeze(0)
            # shape: latent_dim x 1 x input_dim x input_dim x kernel x kernel
            center3 = center3.view(self.input_dim, self.latent_dim, self.input_dim, center3.shape[-2],
                                   center3.shape[-1]).permute(1, 0, 2, 3, 4).unsqueeze(1)

            sign_prods = torch.sign(center1) * torch.sign(center3)
            # shape: latent_dim.input_dim x latent_dim.input_dim x kernel x kernel
            center2 = self.weight2 * self.center_mask2
            center2 = center2.view(self.latent_dim, self.input_dim, self.latent_dim, self.input_dim,
                                   center2.shape[-2], center2.shape[-1])

            center2 = center2.permute(0, 2, 1, 3, 4, 5)
            center2 = sign_prods * torch.abs(center2)
            center2 = center2.permute(
                0, 2, 1, 3, 4, 5).contiguous().view_as(self.weight2)
            masked_weight2 = (center2 * self.center_mask2 +
                              self.weight2 * (1. - self.center_mask2)) * self.mask2

            shared_t = torch.max(
                torch.abs(self.t), torch.tensor(1e-12, device=z.device))

            kernel_mid_y, kernel_mid_x = masked_weight1.shape[-2] // 2, masked_weight1.shape[-1] // 2
            diag1_share = torch.diagonal(
                masked_weight1[..., kernel_mid_y, kernel_mid_x].view(self.latent_dim, self.input_dim,
                                                                     self.input_dim),
                dim1=-2, dim2=-1)[None, :, :, None, None]

            kernel_mid_y, kernel_mid_x = masked_weight2.shape[-2] // 2, masked_weight2.shape[-1] // 2
            diag2_share = masked_weight2[..., kernel_mid_y, kernel_mid_x].view(self.latent_dim, self.input_dim,
                                                                               self.latent_dim,
                                                                               self.input_dim)
            diag2_share = torch.diagonal(diag2_share.permute(0, 2, 1, 3), dim1=-2,
                                         dim2=-1)  # shape: latent_dim x latent_dim x input_dim
            diag2_share = diag2_share[None, :, :, :, None,
                                      None]  # shape: 1 x latent_dim x latent_dim x input_dim x 1 x 1

            kernel_mid_y, kernel_mid_x = masked_weight3.shape[-2] // 2, masked_weight3.shape[-1] // 2
            diag3_share = masked_weight3[..., kernel_mid_y, kernel_mid_x].view(self.input_dim, self.latent_dim,
                                                                               self.input_dim)
            diag3_share = torch.diagonal(diag3_share.permute(
                1, 0, 2), dim1=-2, dim2=-1)[None, :, :, None, None]

            def value_and_grad(x):
                # shape: B x latent_output . input_dim x img_size x img_size
                latent_output = F.conv2d(
                    x, masked_weight1, bias=self.bias1, padding=self.padding1, stride=1)
                diag1 = self.non_linearity_derivative(latent_output). \
                    view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2], x.shape[-1]) \
                    * diag1_share  # shape: B x latent_dim x input_dim x img_shape x img_shape
                latent_output = self.non_linearity(latent_output)
                latent_output = F.conv2d(latent_output, masked_weight2, bias=self.bias2, padding=self.padding2,
                                         stride=1)
                diag2 = torch.sum(diag2_share * diag1.unsqueeze(1),
                                  dim=2)  # shape: B x latent_dim x input_dim x img_shape x img_shape
                latent_output_derivative = self.non_linearity_derivative(
                    latent_output)
                latent_output = self.non_linearity(latent_output)
                latent_output = F.conv2d(latent_output, masked_weight3, bias=self.bias3, padding=self.padding3,
                                         stride=1)
                diag3 = latent_output_derivative.view(x.shape[0], self.latent_dim, self.input_dim, x.shape[-2],
                                                      x.shape[-1]) \
                    * diag3_share  # shape: B x latent_dim x input_dim x img_shape x img_shape
                # shape: B x input_dim x img_shape x img_shape
                diag = torch.sum(diag2 * diag3, dim=1)
                derivative = diag + shared_t  # shape: B x input_dim x img_shape x img_shape
                # shape: B x input_dim x img_shape x img_shape
                output = latent_output + shared_t * x
                return output, derivative

            if self.type == 'A':
                print("type A")
                x = z / shared_t  # [0,...]
                for _ in tqdm(range(self.config.model.n_iters)):
                    output, grad = value_and_grad(x)
                    x += (z - output) / (self.config.analysis.newton_lr * grad)
                return x

            elif self.type == 'B':
                print("type B")
                x = z / shared_t  # [0,...]
                for _ in tqdm(range(self.config.model.n_iters)):
                    output, grad = value_and_grad(x)
                    x += (z - output) / (self.config.analysis.newton_lr * grad)
                return x


class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size * block_size

    def forward(self, x):
        input = x[0]
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output, x[1]

    def sampling(self, z):
        with torch.no_grad():
            input = z
            output = input.permute(0, 2, 3, 1)
            (batch_size, d_height, d_width, d_depth) = output.size()
            s_depth = int(d_depth / self.block_size_sq)
            s_width = int(d_width * self.block_size)
            s_height = int(d_height * self.block_size)
            t_1 = output.reshape(batch_size, d_height,
                                 d_width, self.block_size_sq, s_depth)
            spl = t_1.split(self.block_size, 3)
            stack = [t_t.reshape(batch_size, d_height, s_width, s_depth)
                     for t_t in spl]
            output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                          s_depth)
            output = output.permute(0, 3, 1, 2)
            return output


class Net(nn.Module):
    # layers latent_dim at each layer
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inplanes = channel = config.data.channels

        image_size = config.data.image_size
        init_zero = False
        init_zero_bound = config.model.zero_init_start

        self.layers = nn.ModuleList()
        cur_layer = 0
        self.n_layers = config.model.n_layers
        subsampling_gap = self.n_layers // (config.model.n_subsampling + 1)
        subsampling_anchors = [subsampling_gap *
                               (i + 1) for i in range(config.model.n_subsampling)]

        latent_size = config.model.latent_size

        for layer_num in range(self.n_layers):
            if layer_num in subsampling_anchors:
                self.layers.append(SpaceToDepth(2))
                channel *= 2 * 2
                image_size = int(image_size / 2)
                latent_size //= 2 * 2
                print('space to depth')

            if cur_layer > init_zero_bound:
                init_zero = True

            shape = (channel, image_size, image_size)
            self.layers.append(
                self._make_layer(shape, 1, latent_size, channel, init_zero))
            print('basic block')

        self.sampling_shape = shape

    def _make_layer(self, shape, block_num, latent_dim, input_dim, init_zero):
        layers = []
        for i in range(0, block_num):
            layers.append(BasicBlock(self.config, shape, latent_dim, type='A', input_dim=input_dim,
                                     init_zero=init_zero))

            layers.append(BasicBlock(self.config, shape, latent_dim, type='B', input_dim=input_dim,
                                     init_zero=init_zero))

        return SequentialWithSampling(*layers)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            x, log_det = layer([x, log_det])

        x = x.reshape(x.shape[0], -1)
        return x, log_det

    def sampling(self, z):
        z = z.view(z.shape[0], *self.sampling_shape)
        with torch.no_grad():
            for layer in reversed(self.layers):
                z = layer.sampling(z)

            return z
