#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional
import torch.nn.functional as F
# from timm.models.layers import trunc_normal_, DropPath
class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels


        pool_stride_lst_wanghe = pool_op_kernel_sizes[:3]
        kernel_size_lst_wanghe = conv_kernel_sizes[:4]
        out_channels_lst_wanghe = []
        for i in range(len(kernel_size_lst_wanghe)):
            out_channels_lst_wanghe.append(min(32*2**(i),320))
        self.trans = Robust_TransUnit(out_channels_lst_wanghe, pool_stride_lst_wanghe, kernel_size_lst_wanghe,conv_op)


        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))

            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_down = final_num_features
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)
        skips = self.trans(skips)
        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp


class Robust_DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size = [3,3,3]):
        super().__init__()
        padding = [it//2 for it in kernel_size]
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size= kernel_size, padding=padding),
            # nn.InstanceNorm3d(out_channels,eps = 1e-5,momentum = 0.1, affine = True),
            nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope = 0.01,inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size = kernel_size, padding=padding),
            # nn.InstanceNorm3d(out_channels,eps = 1e-5,momentum = 0.1, affine = True),
            nn.BatchNorm3d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope = 0.01,inplace=True)
            # nn.ReLU(inplace=True),
    def forward(self, x):
        shortcut = self.shortcut(x)
        return self.relu(self.net(x) + shortcut)
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x
class Robust_DoubleConv2(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size = [3,3,3],drop_path = 0.2, layer_scale_init_value=1e-6):

        super().__init__()
        assert in_channels == out_channels
        self.dwconv = nn.Conv3d(in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels) # depthwise conv
        self.norm = LayerNorm(in_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((in_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels,eps = 1e-5,momentum = 0.1, affine = True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            nn.LeakyReLU(negative_slope = 0.01,inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.InstanceNorm3d(out_channels,eps = 1e-5,momentum = 0.1, affine = True),
            # nn.BatchNorm3d(out_channels),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.LeakyReLU(negative_slope = 0.01,inplace=True)
            # nn.ReLU(inplace=True),
    def forward(self, x):
        shortcut = self.shortcut(x)
        return self.relu(self.net(x) + shortcut)


class DoubleConv1(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels,eps = 1e-5,momentum = 0.1, affine = True),
            nn.LeakyReLU(negative_slope = 0.01,inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_channels,eps = 1e-5,momentum = 0.1, affine = True),
            nn.LeakyReLU(negative_slope = 0.01,inplace=True),
            # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
            # nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Robust_Down2(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,pool_stride,kernel_size):
        super(Robust_Down2,self).__init__()
        self.maxpool_conv = nn.Sequential(
            LayerNorm(in_channels, eps=1e-6, data_format="channels_first"),
            nn.Conv3d(in_channels, out_channels, kernel_size=pool_stride, stride=pool_stride),
            # nn.MaxPool3d(kernel_size=pool_stride,stride= pool_stride),
            Robust_DoubleConv2(out_channels, out_channels,kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class Robust_Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,pool_stride,kernel_size):
        super(Robust_Down,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=pool_stride,stride= pool_stride),
            Robust_DoubleConv(in_channels, out_channels,kernel_size)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
def Robust_pools(pool_stride_lst,grade1,grade2,conv_op):
    if grade1 < grade2:
        temp_stridex = 1
        temp_stridey = 1
        temp_stridez = 1
        for i in range(grade1,grade2):
            temp_stridex *= pool_stride_lst[i][0] 
            temp_stridey *= pool_stride_lst[i][1] 
            if conv_op == nn.Conv3d:
                temp_stridez *= pool_stride_lst[i][2] 
        if conv_op == nn.Conv3d:
            return nn.AvgPool3d([temp_stridex,temp_stridey,temp_stridez])
        else:
            return nn.AvgPool2d([temp_stridex,temp_stridey])

    else:
        return nn.Identity()

class Robust_TransUnit(nn.Module):
    """docstring for TransUnit"""
    def __init__(self, in_channels,pool_stride_lst,kernel_size_lst,conv_op):
        super(Robust_TransUnit, self).__init__()
        self.lenth = len(in_channels)
        self.conv_lst = nn.ModuleList()
        self.pool_lst = nn.ModuleList()
        self.bn_lst = nn.ModuleList()
        self.conv_op = conv_op
        for i in range(self.lenth):
            for j in range(self.lenth):
                padding = [it//2 for it in kernel_size_lst[i]]
                self.conv_lst.append(conv_op(in_channels[i],in_channels[j],kernel_size=kernel_size_lst[i], padding=padding,stride=1))
                self.pool_lst.append(Robust_pools(pool_stride_lst,i,j,conv_op))
            if conv_op == nn.Conv3d:
                self.bn_lst.append(nn.BatchNorm3d(in_channels[i]))
            else:
                self.bn_lst.append(nn.BatchNorm2d(in_channels[i]))

    def call_trans(self,x_group,tgt_grade):
        sumx = None
        for i in range(self.lenth):
            if i < tgt_grade:
                tempx = self.pool_lst[i * self.lenth + tgt_grade](x_group[i])
                tempx = self.conv_lst[i * self.lenth + tgt_grade](tempx)
            elif i == tgt_grade:
                tempx = self.conv_lst[i * self.lenth + tgt_grade](x_group[i])
            else:
                tempx = self.conv_lst[i * self.lenth + tgt_grade](x_group[i])
                if self.conv_op == nn.Conv3d:
                    tempx = F.interpolate(tempx,size= x_group[tgt_grade].size()[2:],mode= 'trilinear',align_corners= True)
                else:
                    tempx = F.interpolate(tempx,size= x_group[tgt_grade].size()[2:],mode= 'bilinear',align_corners= True)
            if sumx == None:
                sumx = tempx
            else:
                sumx += tempx
        sumx = F.relu(self.bn_lst[tgt_grade](sumx),inplace= True)
        return sumx
    def forward(self,x_group):
        new_x_group = []
        for i in range(len(x_group)):
            if i < self.lenth:
                # print(x_group[i].shape)
                new_x_group.append(self.call_trans(x_group[:self.lenth],i))
            else:
                new_x_group.append(x_group[i])
        return new_x_group


class TransUnit(nn.Module):
    """docstring for TransUnit"""
    def __init__(self, in_channels):
        super(TransUnit, self).__init__()
        self.in_channels = in_channels
        self.sum_channels = sum(in_channels)
        self.down = nn.AvgPool3d(2)
        self.up = nn.Upsample(scale_factor=2,mode= 'trilinear',align_corners=True)
        self.conv_0_0 = nn.Conv3d(in_channels[0], in_channels[0],kernel_size=3,padding=1,stride=1)
        self.conv_0_1 = nn.Conv3d(in_channels[0], in_channels[1],kernel_size=3,padding=1,stride=1)
        self.conv_0_2 = nn.Conv3d(in_channels[0], in_channels[2],kernel_size=3,padding=1,stride=1)
        self.conv_0_3 = nn.Conv3d(in_channels[0], in_channels[3],kernel_size=3,padding=1,stride=1)

        self.conv_1_0 = nn.Conv3d(in_channels[1], in_channels[0],kernel_size=3,padding=1,stride=1)
        self.conv_1_1 = nn.Conv3d(in_channels[1], in_channels[1],kernel_size=3,padding=1,stride=1)
        self.conv_1_2 = nn.Conv3d(in_channels[1], in_channels[2],kernel_size=3,padding=1,stride=1)
        self.conv_1_3 = nn.Conv3d(in_channels[1], in_channels[3],kernel_size=3,padding=1,stride=1)

        self.conv_2_0 = nn.Conv3d(in_channels[2], in_channels[0],kernel_size=3,padding=1,stride=1)
        self.conv_2_1 = nn.Conv3d(in_channels[2], in_channels[1],kernel_size=3,padding=1,stride=1)
        self.conv_2_2 = nn.Conv3d(in_channels[2], in_channels[2],kernel_size=3,padding=1,stride=1)
        self.conv_2_3 = nn.Conv3d(in_channels[2], in_channels[3],kernel_size=3,padding=1,stride=1)

        self.conv_3_0 = nn.Conv3d(in_channels[3], in_channels[0],kernel_size=3,padding=1,stride=1)
        self.conv_3_1 = nn.Conv3d(in_channels[3], in_channels[1],kernel_size=3,padding=1,stride=1)
        self.conv_3_2 = nn.Conv3d(in_channels[3], in_channels[2],kernel_size=3,padding=1,stride=1)
        self.conv_3_3 = nn.Conv3d(in_channels[3], in_channels[3],kernel_size=3,padding=1,stride=1)

        self.bn0 = nn.BatchNorm3d(in_channels[0])
        self.bn1 = nn.BatchNorm3d(in_channels[1])
        self.bn2 = nn.BatchNorm3d(in_channels[2])
        self.bn3 = nn.BatchNorm3d(in_channels[3])

        self.relu0 = nn.ReLU(inplace= True)
        self.relu1 = nn.ReLU(inplace= True)
        self.relu2 = nn.ReLU(inplace= True)
        self.relu3 = nn.ReLU(inplace= True)
        # self.conv_0 = nn.Sequential(nn.Conv3d(in_channels[0], in_channels[0],kernel_size=3,padding=1,stride=1),
        #                             nn.BatchNorm3d(in_channels[0]),
        #                             nn.ReLU(inplace = True))
        # self.conv_1 = nn.Sequential(nn.Conv3d(in_channels[1], in_channels[1],kernel_size=3,padding=1,stride=1),
        #                             nn.BatchNorm3d(in_channels[1]),
        #                             nn.ReLU(inplace = True))
        # self.conv_2 = nn.Sequential(nn.Conv3d(in_channels[2], in_channels[2],kernel_size=3,padding=1,stride=1),
        #                             nn.BatchNorm3d(in_channels[2]),
        #                             nn.ReLU(inplace = True))
        # self.conv_3 = nn.Sequential(nn.Conv3d(in_channels[3], in_channels[3],kernel_size=3,padding=1,stride=1),
        #                             nn.BatchNorm3d(in_channels[3]),
        #                             nn.ReLU(inplace = True))
        # self.conv_lvl0 = nn.Sequential(
        #   nn.Conv3d(in_channels = self.sum_channels, out_channels= in_channels[0],kernel_size=3,stride=1,padding=1),
        #   nn.BatchNorm3d(in_channels[0]),
        #   nn.ReLU(inplace=True),
        # )
        # self.conv_lvl1 = nn.Sequential(
        #   nn.Conv3d(in_channels = self.sum_channels, out_channels= in_channels[1],kernel_size=3,stride=1,padding=1),
        #   nn.BatchNorm3d(in_channels[1]),
        #   nn.ReLU(inplace=True),
        # )
        # self.conv_lvl2 = nn.Sequential(
        #   nn.Conv3d(in_channels = self.sum_channels, out_channels= in_channels[2],kernel_size=3,stride=1,padding=1),
        #   nn.BatchNorm3d(in_channels[2]),
        #   nn.ReLU(inplace=True),
        # )
        # self.conv_lvl3 = nn.Sequential(
        #   nn.Conv3d(in_channels = self.sum_channels, out_channels= in_channels[3],kernel_size=3,stride=1,padding=1),
        #   nn.BatchNorm3d(in_channels[3]),
        #   nn.ReLU(inplace=True),
        # )     
    def trans_lvl1(self,x_group):
        x1_ = self.conv_0_0(x_group[0])
        x2_ = self.up(self.conv_1_0(x_group[1]))
        x3_ = self.up(self.up(self.conv_2_0(x_group[2])))
        x4_ = self.up(self.up(self.up(self.conv_3_0(x_group[3]))))
        x = x1_ + x2_ + x3_ + x4_
        x = self.relu0(self.bn0(x))
        # x = self.conv_0(x)
        return x
        # return self.relu0(self.bn0(x))
    def trans_lvl2(self,x_group):
        x1_ = self.conv_0_1(self.down(x_group[0]))
        x2_ = self.conv_1_1(x_group[1])
        x3_ = self.up(self.conv_2_1(x_group[2]))
        x4_ = self.up(self.up(self.conv_3_1(x_group[3])))
        x = x1_ + x2_ + x3_ + x4_
        x = self.relu1(self.bn1(x))
        # x = self.conv_1(x)
        # x = torch.cat([x1_,x2_,x3_,x4_],dim=1)
        # x = self.conv_lvl1(x)
        return x
        # return self.relu1(self.bn1(x))
    def trans_lvl3(self,x_group):
        x1_ = self.conv_0_2(self.down(self.down(x_group[0])))
        x2_ = self.conv_1_2(self.down(x_group[1]))
        x3_ = self.conv_2_2(x_group[2])
        x4_ = self.up(self.conv_3_2(x_group[3]))
        x = x1_ + x2_ + x3_ + x4_
        x = self.relu2(self.bn2(x))
        # x = self.conv_2(x)
        # x = torch.cat([x1_,x2_,x3_,x4_],dim=1)
        # x = self.conv_lvl2(x)
        return x
        # return self.relu2(self.bn2(x))
    def trans_lvl4(self,x_group):
        x1_ = self.conv_0_3(self.down(self.down(self.down(x_group[0]))))
        x2_ = self.conv_1_3(self.down(self.down(x_group[1])))
        x3_ = self.conv_2_3(self.down(x_group[2]))
        x4_ = self.conv_3_3(x_group[3])
        x = x1_ + x2_ + x3_ + x4_
        x = self.relu3(self.bn3(x))
        # x = self.conv_3(x)
        # x = torch.cat([x1_,x2_,x3_,x4_],dim=1)
        # x = self.conv_lvl3(x)
        return x
        # return self.relu3(self.bn3(x))
    def forward(self,x_group):
        x1 = self.trans_lvl1(x_group)
        x2 = self.trans_lvl2(x_group)
        x3 = self.trans_lvl3(x_group)
        x4 = self.trans_lvl4(x_group)
        return tuple([x1,x2,x3,x4])

class Robust_ResEncoder(nn.Module):
    def __init__(self, input_channels,out_channels_lst,pool_stride_lst,kernel_size_lst):
        super().__init__()
        self.lenth = len(out_channels_lst)
        self.down_lst = nn.ModuleList()
        # dp_rates=[x.item() for x in torch.linspace(0, 0.2, sum(depths))] 
        for i in range(len(out_channels_lst)):
            if i == 0:
                self.down_lst.append(Robust_DoubleConv(input_channels,out_channels_lst[i],kernel_size=kernel_size_lst[i]))
            else:
                self.down_lst.append(Robust_Down(out_channels_lst[i-1],out_channels_lst[i],pool_stride=pool_stride_lst[i-1],kernel_size=kernel_size_lst[i]))

    def forward(self, x):
        new_x_group = []
        for i in range(self.lenth):
            x = self.down_lst[i](x) 
            tempx = x + 0
            new_x_group.append(tempx)
        return new_x_group

class ResEncoder(nn.Module):
    def __init__(self, input_channels,dim):
        super().__init__()

        self.inc = DoubleConv(input_channels, dim)
        self.down1 = Down(dim, dim*2)
        self.down2 = Down(dim*2, dim*4)
        self.down3 = Down(dim*4, dim*8)
    def forward(self, x):
        x1 = self.inc(x)
        # x1 = self.HR_up(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        return [x1,x2,x3,x4]

class Robust_Out_Fusion(nn.Module):
    """docstring for OutConv"""
    def __init__(self, out_channels_lst,n_classes,pool_stride_lst,kernel_size_lst):
        super(Robust_Out_Fusion, self).__init__()
        self.lenth = len(out_channels_lst)
        self.upsamp = nn.ModuleList()
        self.out_conv = nn.ModuleList()
        for i in range(self.lenth):
            conv_pad = [it//2 for it in kernel_size_lst[i]]
            if i == 0:
                self.out_conv.append(nn.Conv3d(out_channels_lst[self.lenth-1-i],n_classes,kernel_size= kernel_size_lst[i], stride=1, padding= conv_pad))
            else:
                padding = [it//2 for it in pool_stride_lst[self.lenth-1-i]]
                pad_kernel = [(it*3-2) for it in pool_stride_lst[self.lenth-1-i]] # 
                self.upsamp.append(nn.ConvTranspose3d(n_classes,n_classes,kernel_size=pad_kernel,stride=pool_stride_lst[self.lenth-1-i],padding=padding))
                self.out_conv.append(nn.Conv3d(out_channels_lst[self.lenth-1-i]+n_classes,n_classes,kernel_size= kernel_size_lst[i], stride=1, padding= conv_pad))
    def forward(self,x_group):
        new_x_group = []
        for i in range(self.lenth):
            if i == 0:
                tempx = self.out_conv[i](x_group[self.lenth-1-i])
            else:
                tempx = torch.cat([tempx,x_group[self.lenth-1-i]],dim =1)
                tempx = self.out_conv[i](tempx)
            new_x_group.append(tempx)
            if i == self.lenth -1:
                return list(reversed(new_x_group[1:]))
            tempx = self.upsamp[i](tempx)

def exam_out(x_group):
    for i in range(len(x_group)):
        print('!!!',i,x_group[i].shape)
class Out_Fusion(nn.Module):
    """docstring for OutConv"""
    def __init__(self, in_channels,n_classes):
        super(Out_Fusion, self).__init__()
        self.upsamp4to3 = nn.ConvTranspose3d(n_classes,n_classes,kernel_size = 4, stride = 2, padding = 1)
        self.upsamp3to2 = nn.ConvTranspose3d(n_classes,n_classes,kernel_size = 4, stride = 2, padding = 1)
        self.upsamp2to1 = nn.ConvTranspose3d(n_classes,n_classes,kernel_size = 4, stride = 2, padding = 1)


        self.conv_lvl0 = nn.Conv3d(in_channels[0]+n_classes,n_classes,stride=1,kernel_size = 3, padding= 1)
        self.conv_lvl1 = nn.Conv3d(in_channels[1]+n_classes,n_classes,stride=1,kernel_size = 3, padding= 1)
        self.conv_lvl2 = nn.Conv3d(in_channels[2]+n_classes,n_classes,stride=1,kernel_size = 3, padding= 1)
        self.conv_lvl3 = nn.Conv3d(in_channels[3],n_classes,stride=1,kernel_size = 3, padding= 1)


    def forward(self,x_group):

        x4 = self.conv_lvl3(x_group[3])
        x4to3 = self.upsamp4to3(x4)

        x3 = torch.cat([x4to3,x_group[2]],dim = 1)
        x3 = self.conv_lvl2(x3)
        x3to2 = self.upsamp3to2(x3)

        x2 = torch.cat([x3to2,x_group[1]],dim = 1)
        x2 = self.conv_lvl1(x2)
        x2to1 = self.upsamp2to1(x2)

        x1 = torch.cat([x2to1,x_group[0]],dim = 1)
        x1 = self.conv_lvl0(x1)

        # x1 = self.conv_lvl0(x_group[0])
        # x2 = self.conv_lvl1(x_group[1])
        # x3 = self.conv_lvl2(x_group[2])
        # x4 = self.conv_lvl3(x_group[3])
        return [x1,x2,x3,x4]
class Robust_FuFsion_Module(nn.Module):
    def __init__(self, dim_lst,kernel_size_lst):
        super().__init__()
        self.fus_lst = nn.ModuleList()
        self.lenth = len(dim_lst)
        for i in range(self.lenth):
            padding = [it//2 for it in kernel_size_lst[i]]
            self.fus_lst.append(nn.Sequential(nn.Conv3d(2*dim_lst[i],dim_lst[i],kernel_size = kernel_size_lst[i], padding = padding, stride = 1), 
                    nn.BatchNorm3d(dim_lst[i]),
                    nn.ReLU(inplace = True))
            )
    def forward(self,x_group1,x_group2):
        new_x_group = []
        for i in range(self.lenth):
            tempx = torch.cat([x_group1[i],x_group2[i]],dim = 1)
            new_x_group.append(self.fus_lst[i](tempx))
        return new_x_group


class Fusion_Module(nn.Module):
    def __init__(self, dim,dim2 =None):
        super().__init__()
        if dim2 == None:
            dim2 = dim
        dim_sum = dim + dim2

        self.fus1 = nn.Sequential(nn.Conv3d(dim_sum*1,dim,kernel_size = 3, padding = 1, stride = 1), 
                    nn.BatchNorm3d(dim*1),
                    nn.ReLU(inplace = True))

        self.fus2 = nn.Sequential(nn.Conv3d(dim_sum*2,dim*2,kernel_size = 3, padding = 1, stride = 1), 
                    nn.BatchNorm3d(dim*2),
                    nn.ReLU(inplace = True))

        self.fus3 = nn.Sequential(nn.Conv3d(dim_sum*4,dim*4,kernel_size = 3, padding = 1, stride = 1), 
                    nn.BatchNorm3d(dim*4),
                    nn.ReLU(inplace = True))
        self.fus4 = nn.Sequential(nn.Conv3d(dim_sum*8,dim*8,kernel_size = 3, padding = 1, stride = 1), 
                    nn.BatchNorm3d(dim*8),
                    nn.ReLU(inplace = True))
    def forward(self,x_group1,x_group2):
        x1 = torch.cat([x_group1[0],x_group2[0]],dim = 1)
        x1 = self.fus1(x1)
        x2 = torch.cat([x_group1[1],x_group2[1]],dim = 1)
        x2 = self.fus2(x2)
        x3 = torch.cat([x_group1[2],x_group2[2]],dim = 1)
        x3 = self.fus3(x3)
        x4 = torch.cat([x_group1[3],x_group2[3]],dim = 1)
        x4 = self.fus4(x4)
        return [x1,x2,x3,x4]

