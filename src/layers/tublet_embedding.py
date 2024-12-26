import torch.nn as nn
from einops.layers.torch import Rearrange

class TubletEmbedding(nn.Module):
    """
    "We consider two simple methods for mapping a video V ∈ RT ×H×W ×C to a sequence of tokens z̃ ∈ Rnt ×nh ×nw ×d.
    We then add the positional embedding and N ×dreshape into R to obtain z, the input to the transformer."

    "Extract non-overlapping, spatio-temporal “tubes” from the input volume, and to linearly project this to Rd.
    For a tubelet of di- mension t × h × w, nt = b Tt c, nh = b H h c and nw = b W w c, tokens are extracted from the temporal, height, and width dimensions respectively.
    Smaller tubelet dimensions thus result in more tokens which increases the computation.
    This method fuses spatio-temporal information during tokenisation, in contrast to “Uniform frame sam- pling” where temporal information
    from different frames is fused by the transformer.

    We denote as 'central frame initialisation', where E is initialised with zeroes
    along all temporal positions, except at the centre b t c, 2E = [0, . . . , Eimage, . . . , 0]
    """
    def __init__(self, vit, tempo_patch_size, spat_patch_size, hidden_dim):
        super().__init__()

        kernel_size = stride = (
            tempo_patch_size, spat_patch_size, spat_patch_size,
        ) # "$t \times h \times h$"
        self.conv3d = nn.Conv3d(3, hidden_dim, kernel_size, stride, 0)
        self.conv3d.bias.data = vit.patch_embed.proj.bias.data
        self.to_seq = Rearrange("b d t h w -> b t (h w) d")

    def forward(self, x):
        """
        Args:
        x (torch.Tensor): Input tensor of shape (B, C, T, H, W).
        """
        x = self.conv3d(x) # (B, d, $n_{t}$, $n_{h}$, $n_{w}$)
        return self.to_seq(x)
