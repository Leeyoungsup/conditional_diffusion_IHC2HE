from abc import abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Upsample(nn.Module):
    """Upsampling layer with optional convolution"""
    def __init__(self, in_ch: int, out_ch: int, with_conv: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """Downsampling layer"""
    def __init__(self, in_ch: int, out_ch: int, with_conv: bool = True):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        else:
            self.conv = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class EmbedBlock(nn.Module):
    """Abstract class for blocks that use time/condition embeddings"""
    @abstractmethod
    def forward(self, x, temb, cemb):
        pass


class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x: torch.Tensor, temb: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x


class ResBlock(EmbedBlock):
    """
    ResBlock with AdaGN (Adaptive Group Normalization)
    """
    def __init__(self, in_ch: int, out_ch: int, tdim: int, cdim: int, droprate: float = 0.1):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # First normalization and convolution
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        
        # Time and condition embedding projections
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch * 2),
        )
        self.cemb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cdim, out_ch * 2),
        )
        
        # Second normalization and convolution
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout(p=droprate)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        
        # Residual connection
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        h = x
        
        # First conv block
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)
        
        # Apply adaptive normalization
        temb_out = self.temb_proj(temb)[:, :, None, None]
        cemb_out = self.cemb_proj(cemb)[:, :, None, None]
        
        # Split into scale and shift
        t_scale, t_shift = temb_out.chunk(2, dim=1)
        c_scale, c_shift = cemb_out.chunk(2, dim=1)
        
        # Apply AdaGN
        h = h * (1 + t_scale + c_scale) + (t_shift + c_shift)
        
        # Second conv block
        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class EfficientAttention(nn.Module):
    """Memory-efficient attention"""
    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        self.in_ch = in_ch
        self.num_heads = num_heads
        self.head_dim = in_ch // num_heads
        assert in_ch % num_heads == 0, "in_ch must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(min(32, in_ch), in_ch)
        self.qkv = nn.Conv2d(in_ch, in_ch * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Skip attention for large feature maps
        if H * W > 1024:
            return x
        
        h = self.norm(x)
        qkv = self.qkv(h)
        
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        out = self.proj_out(out)
        return x + out


class AttnBlock(nn.Module):
    """Wrapper for attention block"""
    def __init__(self, in_ch: int, num_heads: int = 4):
        super().__init__()
        while in_ch % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attention = EfficientAttention(in_ch, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(x)


class ImprovedUnet(nn.Module):
    """
    FIXED VERSION - Memory-efficient U-Net for conditional diffusion
    """
    def __init__(
        self, 
        in_ch=3, 
        mod_ch=64,
        out_ch=3, 
        ch_mul=[1, 2, 4, 8],
        num_res_blocks=2, 
        cdim=10, 
        use_conv=True, 
        droprate=0.1,
        num_heads=4,
        use_checkpoint=False,
        dtype=torch.float32
    ):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.dtype = dtype
        self.use_checkpoint = use_checkpoint
        
        # Time and condition embedding dimensions
        tdim = mod_ch * 4
        
        # Time embedding layers
        self.temb_layer = nn.Sequential(
            nn.Linear(mod_ch, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        
        # Condition embedding layers
        self.cemb_layer = nn.Sequential(
            nn.Linear(cdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        
        # ⭐ FIXED: Downsampling blocks (중복 선언 제거)
        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, mod_ch, kernel_size=3, padding=1))
        ])
        
        now_ch = mod_ch
        chs = [now_ch]
        
        for i, mul in enumerate(ch_mul):
            nxt_ch = mul * mod_ch
            for _ in range(num_res_blocks):
                layers = [ResBlock(now_ch, nxt_ch, tdim, tdim, droprate)]
                # Only add attention at the lowest resolution
                if i == len(ch_mul) - 1:
                    layers.append(AttnBlock(nxt_ch, num_heads))
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            
            if i != len(ch_mul) - 1:
                self.downblocks.append(
                    EmbedSequential(Downsample(now_ch, now_ch, use_conv))
                )
                chs.append(now_ch)
        
        # Middle blocks
        self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, tdim, droprate),
            AttnBlock(now_ch, num_heads),
            ResBlock(now_ch, now_ch, tdim, tdim, droprate),
        )
        
        # Upsampling blocks
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(ch_mul))[::-1]:
            nxt_ch = mul * mod_ch
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(now_ch + chs.pop(), nxt_ch, tdim, tdim, droprate)]
                if i == len(ch_mul) - 1:
                    layers.append(AttnBlock(nxt_ch, num_heads))
                now_ch = nxt_ch
                
                if i > 0 and j == num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch, use_conv))
                
                self.upblocks.append(EmbedSequential(*layers))
        
        # Output layers
        self.out = nn.Sequential(
            nn.GroupNorm(min(32, now_ch), now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, out_ch, kernel_size=3, padding=1),
        )
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ⭐ FIXED: forward 메서드를 클래스 내부의 올바른 위치로 이동
    def forward(self, x: torch.Tensor, t: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: Input tensor [B, C, H, W]
            t: Timestep tensor [B]
            cemb: Condition embedding [B, cdim]
        """
        # Generate embeddings
        temb = self.temb_layer(timestep_embedding(t, self.mod_ch))
        cemb = self.cemb_layer(cemb)
        
        # Downsampling path
        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, temb, cemb, use_reentrant=False)
            else:
                h = block(h, temb, cemb)
            hs.append(h)
        
        # Middle blocks
        if self.use_checkpoint and self.training:
            h = torch.utils.checkpoint.checkpoint(self.middleblocks, h, temb, cemb, use_reentrant=False)
        else:
            h = self.middleblocks(h, temb, cemb)
        
        # Upsampling path
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim=1)
            if self.use_checkpoint and self.training:
                h = torch.utils.checkpoint.checkpoint(block, h, temb, cemb, use_reentrant=False)
            else:
                h = block(h, temb, cemb)
        
        # Output
        h = h.type(self.dtype)
        return self.out(h)


class ImprovedUnetWithMask(nn.Module):
    """Memory-efficient U-Net with mask input"""
    def __init__(
        self, 
        in_ch=3, 
        mask_ch=5, 
        mod_ch=64,
        out_ch=3, 
        ch_mul=[1, 2, 4, 8], 
        num_res_blocks=2, 
        cdim=10, 
        use_conv=True, 
        droprate=0.1,
        num_heads=4,
        use_checkpoint=False,
        dtype=torch.float32
    ):
        super().__init__()
        self.base_model = ImprovedUnet(
            in_ch=in_ch + mask_ch,
            mod_ch=mod_ch,
            out_ch=out_ch,
            ch_mul=ch_mul,
            num_res_blocks=num_res_blocks,
            cdim=cdim,
            use_conv=use_conv,
            droprate=droprate,
            num_heads=num_heads,
            use_checkpoint=use_checkpoint,
            dtype=dtype
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor, t: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        x_with_mask = torch.cat([x, mask], dim=1)
        return self.base_model(x_with_mask, t, cemb)


# Lightweight version for extreme memory constraints
class LightweightUnet(nn.Module):
    """Ultra-lightweight U-Net (NO ATTENTION)"""
    def __init__(
        self, 
        in_ch=3, 
        mod_ch=32,
        out_ch=3, 
        ch_mul=[1, 2, 4, 4],
        num_res_blocks=1,
        cdim=10, 
        use_conv=True, 
        droprate=0.0,
        dtype=torch.float32
    ):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.dtype = dtype
        
        tdim = mod_ch * 4
        
        self.temb_layer = nn.Sequential(
            nn.Linear(mod_ch, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        
        self.cemb_layer = nn.Sequential(
            nn.Linear(cdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        
        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, mod_ch, kernel_size=3, padding=1))
        ])
        
        now_ch = mod_ch
        chs = [now_ch]
        
        for i, mul in enumerate(ch_mul):
            nxt_ch = mul * mod_ch
            for _ in range(num_res_blocks):
                layers = [ResBlock(now_ch, nxt_ch, tdim, tdim, droprate)]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            
            if i != len(ch_mul) - 1:
                self.downblocks.append(
                    EmbedSequential(Downsample(now_ch, now_ch, use_conv))
                )
                chs.append(now_ch)
        
        self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, tdim, droprate),
            ResBlock(now_ch, now_ch, tdim, tdim, droprate),
        )
        
        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(ch_mul))[::-1]:
            nxt_ch = mul * mod_ch
            for j in range(num_res_blocks + 1):
                layers = [ResBlock(now_ch + chs.pop(), nxt_ch, tdim, tdim, droprate)]
                now_ch = nxt_ch
                
                if i > 0 and j == num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch, use_conv))
                
                self.upblocks.append(EmbedSequential(*layers))
        
        self.out = nn.Sequential(
            nn.GroupNorm(min(32, now_ch), now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        temb = self.temb_layer(timestep_embedding(t, self.mod_ch))
        cemb = self.cemb_layer(cemb)
        
        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h, temb, cemb)
            hs.append(h)
        
        h = self.middleblocks(h, temb, cemb)
        
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, temb, cemb)
        
        h = h.type(self.dtype)
        return self.out(h)