from src.models.base_model import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict

from src.layers.block import Block
from src.layers.patch_embedding import PatchEmbed


class VisionTransformer(BaseModel):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self,
                 img_size=400,
                 patch_size=16,
                 in_chans=9,
                 num_classes=1,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None
                 ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of output channels for segmentation mask
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.fire_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.landscape_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.fire_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Final convolution for segmentation mask
        self.segmentation_head = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.segmentation_head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.segmentation_head = nn.Conv2d(self.embed_dim, num_classes, kernel_size=1)

    def forward_features(self, x):
        """
        Args:
            x: Tuple containing:
               - fire_state: Tensor of shape (batch_size, fire_frame_index, bitmask, height, width)
               - landscape_features: Tensor of shape (batch_size, 8, height, width)
        """
        x1, x2 = x

        # Shape: (batch_size, fire_frame_index, bitmask, height, width)
        B, T, C1, H, W = x1.shape
        # Shape: (batch_size, fire_frame_index * bitmask, height, width)
        x1 = x1.view(B, -1, H, W)

        # Patch embedding
        x1 = self.fire_embed(x1)  # (batch_size, num_patches, embed_dim)
        x2 = self.landscape_embed(x2)  # (batch_size, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 1:]  # Exclude cls token

    def forward(self, x):
        x = self.forward_features(x)  # (batch_size, num_patches, embed_dim)
        B, _, H, W = x.shape
        x = x.transpose(1, 2).reshape(B, self.embed_dim, H // self.patch_embed.patch_size, W // self.patch_embed.patch_size)
        x = self.segmentation_head(x)  # (batch_size, num_classes, H, W)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)  # Upsample to input size
        return x

