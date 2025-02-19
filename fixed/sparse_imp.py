import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedAttention(nn.Module):
    def __init__(self, heads, n_ctx, fixed_pattern):
        super(FixedAttention, self).__init__()
        self.heads = heads
        self.n_ctx = n_ctx
        self.fixed_pattern = fixed_pattern
        self.register_buffer('attention_mask', self._create_fixed_attention_mask())

    def _create_fixed_attention_mask(self):
        mask = torch.zeros(self.n_ctx, self.n_ctx)
        for i, pattern in enumerate(self.fixed_pattern):
            mask[i, pattern] = 1
        return mask.unsqueeze(0).unsqueeze(0).expand(self.heads, -1, -1, -1)

    def forward(self, q, k, v):
        b, t, e = q.size()
        h = self.heads
        d = e // h

        q = q.view(b, t, h, d).transpose(1, 2)
        k = k.view(b, t, h, d).transpose(1, 2)
        v = v.view(b, t, h, d).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        scores = scores.masked_fill(self.attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, e)
        return out

if __name__ == "__main__":
    n_batch = 4
    n_ctx = 1024
    n_embd = 256
    heads = 4
    attn_mode = "all"
    local_attn_ctx = 32
    blocksize = 32

    # Set a seed for reproducibility
    torch.manual_seed(0)

    q = torch.randn(n_batch, n_ctx, n_embd)
    k = torch.randn(n_batch, n_ctx, n_embd)
    v = torch.randn(n_batch, n_ctx, n_embd)

    model = FixedAttention(heads, n_ctx)
    output = model(q, k, v)
    
    # Print the first few elements of the first batch
    print(output[0, :5, :5])