import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
import math



class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
       
    def forward(self, x):
        # The detailed code will be released after the manuscript is accepted.
        return x



###############################################################################
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()


    def forward(self, x):
        # The detailed code will be released after the manuscript is accepted.
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

    def forward(self, x):
        # The detailed code will be released after the manuscript is accepted.
        return x


class CAAF(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CAAF, self).__init__()

    def forward(self, rgb, depth):
        # The detailed code will be released after the manuscript is accepted.

        return fusion


class SwinTransformerSys(nn.Module):
    def __init__(self, img_size=320, patch_size=4, in_chans=4, num_classes=1000,
                 embed_dim=128, depths=[2, 2, 2], depths_3d=[2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=10, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", **kwargs):
        super().__init__()

        print(
            "SwinTransformerSys expand initial----depths:{};depths_decoder:{};drop_path_rate:{};num_classes:{}".format(
                depths,
                depths_decoder, drop_path_rate, num_classes))

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.patch_embed_depth = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
            self.absolute_pos_embed_depth = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed_depth, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.pos_drop_depth = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build encoder and bottleneck layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.layers_depth = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer_depth = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                     input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                       patches_resolution[1] // (2 ** i_layer)),
                                     depth=depths_3d[i_layer],
                                     num_heads=num_heads[i_layer],
                                     window_size=window_size,
                                     mlp_ratio=self.mlp_ratio,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     drop=drop_rate, attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(depths_3d[:i_layer]):sum(depths_3d[:i_layer + 1])],
                                     norm_layer=norm_layer,
                                     downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                     use_checkpoint=use_checkpoint)
            self.layers_depth.append(layer_depth)

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2 * int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                      int(embed_dim * 2 ** (
                                                  self.num_layers - 1 - i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer == 0:
                layer_up = PatchExpand(
                    input_resolution=(patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)), dim_scale=2, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                         input_resolution=(
                                         patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                         patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                         depth=depths[(self.num_layers - 1 - i_layer)],
                                         num_heads=num_heads[(self.num_layers - 1 - i_layer)],
                                         window_size=window_size,
                                         mlp_ratio=self.mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop_rate, attn_drop=attn_drop_rate,
                                         drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                             depths[:(self.num_layers - 1 - i_layer) + 1])],
                                         norm_layer=norm_layer,
                                         upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, img, depth):

        # The detailed code will be released after the manuscript is accepted.
        return x, img_feats, depth_feats, fuse_feats


    def forward_up_features_fusion(self, x, img_feats):
        # The detailed code will be released after the manuscript is accepted.

        return x


    def fuse_swinUP(self, x):
       # The detailed code will be released after the manuscript is accepted.
        return x

    def fuse_swinUP_L1(self, x):
       # The detailed code will be released after the manuscript is accepted.
        return x
    def fuse_swinUP_L2(self, x):
      # The detailed code will be released after the manuscript is accepted.
        return x
    def fuse_swinUP_L3(self, x):
       # The detailed code will be released after the manuscript is accepted.
        return x

    def img_swinUP(self, x):
      # The detailed code will be released after the manuscript is accepted.
        return x

    def dep_swinUP(self, x):
       # The detailed code will be released after the manuscript is accepted.
        return x


    def PCM_L1(self, img_cam, dep_cam):
       # The detailed code will be released after the manuscript is accepted.
        return img_cam_rv, dep_cam_rv

    def PCM_L2(self, img_cam, dep_cam):
        # The detailed code will be released after the manuscript is accepted.
        return img_cam_rv, dep_cam_rv


    def forward(self, x, depth):
        # The detailed code will be released after the manuscript is accepted.
        return (seg_pre_img, seg_pre_dep, seg_pre_fusion, hidden_feats_seg)

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class Swin_RaiSDNet(nn.Module):
    def __init__(self, img_size=256, in_chans=3, num_classes=21843, zero_head=False):
        super(Swin_RaiSDNet, self).__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.in_chans = in_chans

        self.swin_raisdnet = SwinTransformerSys(img_size=320,
                                                   patch_size=4,
                                                   in_chans=self.in_chans,
                                                   num_classes=2,
                                                   window_size=10,
                                                   embed_dim=96,
                                                   depths=[2, 2, 6],
                                                   depths_3d=[2, 2, 2],
                                                   num_heads=[3,6,12],
                                                   mlp_ratio=4.,
                                                   qkv_bias=True,
                                                   qk_scale=None,
                                                   drop_rate=0,
                                                   drop_path_rate=0.2,
                                                   ape=False,
                                                   patch_norm=True,
                                                   use_checkpoint=False)

    def forward(self, x, depth):
        seg_pre_img, seg_pre_dep, seg_pre_fusion, hidden_feats_seg = self.swin_raisdnet(x, depth)
        return seg_pre_img, seg_pre_dep, seg_pre_fusion, hidden_feats_seg


if __name__ == "__main__":
    input = torch.ones(16, 3, 587, 587)
    depth = torch.ones(16, 3, 587, 587)
    net = Swin_RaiSDNet(num_classes=2)
    seg_pre_img, seg_pre_dep, seg_pre_fusion, hidden_feats_seg = net.forward(input, depth)
    print(seg_pre_img.size())

