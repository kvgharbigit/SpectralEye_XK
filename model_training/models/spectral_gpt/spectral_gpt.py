from functools import partial

import torch
import torch.nn as nn
from torch.nn import LayerNorm


def calculate_metrics_per_pixel(original_spectrum, reconstructed_spectrum):
    epsilon = 1e-10  # 避免除零错误

    # 计算光谱角（Spectral Angle）逐像素
    spectral_angle_per_pixel = torch.acos(torch.sum(original_spectrum * reconstructed_spectrum, dim=1) /
                                          (torch.norm(original_spectrum, dim=1) * torch.norm(reconstructed_spectrum,
                                                                                             dim=1) + epsilon))  #
    return spectral_angle_per_pixel


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def forward(self, x):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(keep_prob) * random_tensor


class TransformerBlock(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(
            self,
            img_size: tuple[int, int] = (256, 256),
            patch_size: tuple[int, int] = (16, 16),
            in_chans: int = 1,
            embed_dim: int = 768,
            num_wavelengths: int = 32,
            wavelength_patch_size: int = 4,
    ):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0, "Image dimensions must be divisible by the patch size"
        assert img_size[1] % patch_size[1] == 0, "Image dimensions must be divisible by the patch size"
        assert num_wavelengths % wavelength_patch_size == 0, "Number of wavelengths must be divisible by the patch size"

        kernel_size = (wavelength_patch_size,) + tuple(patch_size)

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)
        return x


class EncoderVit(nn.Module):
    def __init__(
            self,
            img_size=256,
            num_channels=1,
            num_wavelengths=16,
            spatial_patch_size=16,
            wavelength_patch_size=4,
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            mlp_ratio=4.0,
            mask_ratio=0.9,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.num_channels = num_channels
        self.num_wavelengths = num_wavelengths

        self.spatial_patch_size = (spatial_patch_size, spatial_patch_size)
        self.wavelength_patch_size = wavelength_patch_size

        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_num_heads = encoder_num_heads

        self.num_img_patches = (img_size // spatial_patch_size), (img_size // spatial_patch_size)
        self.num_wavelength_patches = num_wavelengths // wavelength_patch_size

        self.total_patches = self.num_img_patches[0] * self.num_img_patches[1] * self.num_wavelength_patches

        self.voxels_per_mini_patch = (self.spatial_patch_size[0] *
                                      self.spatial_patch_size[1] *
                                      self.wavelength_patch_size * num_channels)

        self.mlp_ratio = mlp_ratio
        self.mask_ratio = mask_ratio

        self.encoder_cls_token = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        self.patch_embed = PatchEmbed(
            self.img_size,
            self.spatial_patch_size,
            self.num_channels,
            self.encoder_embed_dim,
            self.num_wavelengths,
            self.wavelength_patch_size,
        )

        self.encoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_img_patches[0] * self.num_img_patches[1], self.encoder_embed_dim)
        )
        self.encoder_pos_embed_wavelength = nn.Parameter(
            torch.zeros(1, self.num_wavelength_patches, self.encoder_embed_dim)
        )
        self.encoder_pos_embed_class = nn.Parameter(torch.zeros(1, 1, self.encoder_embed_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.encoder_embed_dim,
                    self.encoder_num_heads,
                    self.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=partial(LayerNorm, eps=1e-6),
                )
                for i in range(self.encoder_depth)
            ]
        )
        self.norm = LayerNorm(self.encoder_embed_dim)

        self.initialize_weights()

        # self.patch_info = self.set_patch_info()

    def initialize_weights(self):
        torch.nn.init.trunc_normal_(self.encoder_cls_token, std=0.02)

        torch.nn.init.trunc_normal_(self.encoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.encoder_pos_embed_wavelength, std=0.02)
        torch.nn.init.trunc_normal_(self.encoder_pos_embed_class, std=0.02)

        w = self.patch_embed.proj.weight.data

        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def set_patch_info(self):
    #     T = self.num_wavelengths
    #     H = self.img_size[0]
    #     W = self.img_size[1]
    #
    #     p = self.patch_embed.patch_size[0]
    #     u = self.wavelength_patch_size
    #     assert H == W and H % p == 0 and T % u == 0
    #     h = w = H // p
    #     t = T // u
    #
    #     return T, H, W, p, u, t, h, w

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        assert L == self.total_patches, "Input sequence length should be equal to total number of patches"
        assert D == self.encoder_embed_dim, "Input sequence dimension should be equal to embedding dimension of the encoder"

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is removed
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward(self, x, masking=False):
        x = torch.unsqueeze(x, dim=1)

        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        assert T == self.num_wavelength_patches, "Wavelength patches should be equal to number of wavelength patches"
        assert L == self.num_img_patches[0] * self.num_img_patches[
            1], "Spatial patches should be equal to number of spatial patches"
        assert C == self.encoder_embed_dim, "Embedding dimension should be equal to encoder embedding dimension"

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        if masking:
            x, mask, ids_restore, ids_keep = self.random_masking(x, self.mask_ratio)
        # x = x.view(N, -1, C)
        encoder_cls_token = self.encoder_cls_token
        encoder_cls_token = encoder_cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((encoder_cls_token, x), dim=1)

        pos_embed = (self.encoder_pos_embed_spatial.repeat(1, self.num_wavelength_patches, 1)
                     + torch.repeat_interleave(
                    self.encoder_pos_embed_wavelength, self.num_img_patches[0] * self.num_img_patches[1], dim=1))
        pos_embed = pos_embed.expand(x.shape[0], -1, -1)

        if masking:
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
        pos_embed = torch.cat([self.encoder_pos_embed_class.expand(pos_embed.shape[0], -1, -1), pos_embed], 1)

        x = x.view([N, -1, C]) + pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x_cls = x[:, 0, :]
        x = x[:, 1:, :]

        if masking:
            return x, x_cls, mask, ids_restore
        else:
            return x, x_cls


class DecoderVit(nn.Module):
    def __init__(
            self,
            img_size=256,
            num_channels=1,
            num_wavelengths=16,
            spatial_patch_size=16,
            wavelength_patch_size=4,
            encoder_embed_dim=1024,
            decoder_embed_dim=512,
            depth=4,
            num_heads=16,
            mlp_ratio=4.0,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.num_channels = num_channels
        self.num_wavelengths = num_wavelengths

        self.spatial_patch_size = (spatial_patch_size, spatial_patch_size)
        self.wavelength_patch_size = wavelength_patch_size

        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        # self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.num_img_patches = (img_size // spatial_patch_size), (img_size // spatial_patch_size)
        self.num_wavelength_patches = num_wavelengths // wavelength_patch_size

        self.total_patches = self.num_img_patches[0] * self.num_img_patches[1] * self.num_wavelength_patches

        self.voxels_per_mini_patch = (self.spatial_patch_size[0] *
                                      self.spatial_patch_size[1] *
                                      self.wavelength_patch_size * num_channels)

        self.decoder_embed = nn.Linear(self.encoder_embed_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_pos_embed_spatial = nn.Parameter(
            torch.zeros(1, self.num_img_patches[0] * self.num_img_patches[1], self.decoder_embed_dim)
        )
        self.decoder_pos_embed_temporal = nn.Parameter(
            torch.zeros(1, self.num_wavelength_patches, self.decoder_embed_dim)
        )
        # self.decoder_pos_embed_class = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.decoder_embed_dim,
                    self.num_heads,
                    self.mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=partial(LayerNorm, eps=1e-6),
                )
                for i in range(self.depth)
            ]
        )

        self.voxels_per_mini_patch = spatial_patch_size ** 2 * self.wavelength_patch_size * num_channels

        self.decoder_norm = LayerNorm(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.voxels_per_mini_patch, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # torch.nn.init.trunc_normal_(self.decoder_cls_token, std=0.02)

        torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
        torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)
        # torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)

        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # def set_patch_info(self):
    #     T = self.num_wavelengths
    #     H = self.img_size[0]
    #     W = self.img_size[1]
    #
    #     p = self.patch_embed.patch_size[0]
    #     u = self.wavelength_patch_size
    #     assert H == W and H % p == 0 and T % u == 0
    #     h = w = H // p
    #     t = T // u
    #
    #     return T, H, W, p, u, t, h, w

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N = x.shape[0]
        # T, H, W, p, u, t, h, w = self.patch_info
        T = self.num_wavelengths
        H = self.img_size[0]
        W = self.img_size[1]
        p = self.spatial_patch_size[0]
        u = self.wavelength_patch_size
        t = self.num_wavelength_patches
        h = w = self.num_img_patches[0]

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))
        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs

    def forward(self, x, ids_restore):
        N = x.shape[0]
        T = self.num_wavelength_patches
        H = W = self.num_img_patches[0]

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])

        # decoder_cls_token = self.decoder_cls_token
        # decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((decoder_cls_tokens, x), dim=1)

        decoder_pos_embed = (self.decoder_pos_embed_spatial.repeat(1, self.num_wavelength_patches, 1)
                             + torch.repeat_interleave(
                    self.decoder_pos_embed_temporal, self.num_img_patches[0] * self.num_img_patches[1], dim=1, ))

        # decoder_pos_embed = torch.cat([self.decoder_pos_embed_class.expand(decoder_pos_embed.shape[0], -1, -1),
        #                                decoder_pos_embed], 1)

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        # x_cls = x[:, 0, :]
        # x = x[:, 1:, :]
        # return x, x_cls
        return x


class MaskedAutoencoderViT(nn.Module):
    def __init__(
            self,
            img_size=256,
            num_channels=1,
            num_wavelengths=16,
            spatial_patch_size=16,
            wavelength_patch_size=4,
            encoder_embed_dim=1024,
            encoder_depth=24,
            encoder_num_heads=16,
            decoder_embed_dim=512,
            decoder_depth=4,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            mask_ratio=0.9,
    ):
        super().__init__()
        self.encoder = EncoderVit(
            img_size,
            num_channels,
            num_wavelengths,
            spatial_patch_size,
            wavelength_patch_size,
            encoder_embed_dim,
            encoder_depth,
            encoder_num_heads,
            mlp_ratio,
            mask_ratio,
        )

        self.decoder = DecoderVit(
            img_size,
            num_channels,
            num_wavelengths,
            spatial_patch_size,
            wavelength_patch_size,
            encoder_embed_dim,
            decoder_embed_dim,
            decoder_depth,
            decoder_num_heads,
            mlp_ratio,
        )

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N = imgs.shape[0]
        C = self.encoder.num_channels

        # T, H, W, p, u, t, h, w = self.patch_info
        T = self.encoder.num_wavelengths
        H = self.encoder.img_size[0]
        W = self.encoder.img_size[1]
        p = self.encoder.spatial_patch_size[0]
        u = self.encoder.wavelength_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = self.encoder.num_img_patches[0]
        w = self.encoder.num_img_patches[1]
        t = self.encoder.num_wavelength_patches

        x = imgs.reshape(shape=(N, C, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p ** 2 * C))
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        # 维度转换
        # imgs = imgs[:, :-1, :, :]  # 切片处理数据维度
        imgs = torch.unsqueeze(imgs, dim=1)

        target1 = self.patchify(imgs)

        N, C, T, H, W = imgs.shape
        # p = self.encoder.num_img_patches[0]
        p = self.encoder.spatial_patch_size[0]
        u = self.encoder.wavelength_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u
        # print(target1.shape, self.voxels_per_mini_patch)
        target_whole = target1.reshape(N, t, h * w, self.encoder.voxels_per_mini_patch)  # 2,4,256,192
        target_spatial = target_whole.sum(dim=1)  # 2,4,192
        pred_whole = pred.reshape(N, t, h * w, self.encoder.voxels_per_mini_patch)
        pred_spatial = pred_whole.sum(dim=1)

        loss1 = (pred - target1) ** 2  # pred: 2,1024,192
        loss1 = loss1.mean(dim=-1)  # [N, L], mean loss per patch #2,1024
        mask = mask.view(loss1.shape)
        loss1 = (loss1 * mask).sum() / mask.sum()

        loss3 = (pred_spatial - target_spatial) ** 2  # 2,4,192
        loss3 = loss3.mean(dim=-1)
        mask3 = torch.ones([N, h * w], device=loss3.device)
        loss3 = (loss3 * mask3).sum() / mask3.sum()

        loss = loss1 + loss3  # mean loss on removed patches
        return loss

    def forward(self, imgs):
        latent, cls_enc, mask, ids_restore = self.encoder(imgs, masking=True)
        # pred, cls_dec = self.decoder(latent, ids_restore)  # [N, L, p*p*3]
        pred = self.decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch8_256():
    model = MaskedAutoencoderViT(
        img_size=256,
        num_channels=1,
        num_wavelengths=15,
        spatial_patch_size=16,
        wavelength_patch_size=3,
        encoder_embed_dim=768,
        decoder_embed_dim=256,
        encoder_depth=12,
        encoder_num_heads=12,
        mlp_ratio=4,
        mask_ratio=0.9,
    )
    return model


if __name__ == '__main__':
    x = torch.rand(2, 30, 500, 500).to('cuda')
    model = MaskedAutoencoderViT(
        img_size=500,
        num_channels=1,
        num_wavelengths=30,
        spatial_patch_size=25,
        wavelength_patch_size=5,
        encoder_embed_dim=512,
        encoder_depth=8,
        encoder_num_heads=8,
        decoder_embed_dim=256,
        decoder_depth=4,
        decoder_num_heads=4,
        mlp_ratio=4,
        mask_ratio=0.9,
    )

    model = model.to('cuda')

    # model = nn.DataParallel(model, device_ids=[0, 1, 2])
    output = model(x)

    print(output[1].shape)
