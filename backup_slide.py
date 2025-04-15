    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head
        head_dim = C // H

        # Compute query, key, value projections
        q, k, v = self.c_attn(x).split(C, dim=2)

        # Reshape into multi-head format
        q = q.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, hs)
        k = k.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, hs)
        v = v.view(B, T, H, head_dim).transpose(1, 2)  # (B, H, T, hs)

        # Pad sequence length to be divisible by window size
        W = self.window_size
        T_pad = (T + W - 1) // W * W
        pad_length = T_pad - T
        if pad_length > 0:
            q = F.pad(q, (0, 0, 0, pad_length))
            k = F.pad(k, (0, 0, 0, pad_length))
            v = F.pad(v, (0, 0, 0, pad_length))

        # Create sliding window indices
        max_range = torch.arange(T_pad, device=x.device)
        indices = max_range.unsqueeze(1) - torch.arange(W, device=x.device)
        indices = torch.clamp(indices, min=0)

        # Gather sliding window key/value tensors
        k_windows = k[:, :, indices, :]  # (B, H, T_pad, W, hs)
        v_windows = v[:, :, indices, :]  # (B, H, T_pad, W, hs)

        # Compute attention
        q_windows = q.unsqueeze(3)  # (B, H, T_pad, 1, hs)
        att_weights = F.scaled_dot_product_attention(
            q_windows, k_windows, v_windows,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0,
            is_causal=True
        )  # (B, H, T_pad, 1, hs)

        # Remove padding and extra dimension
        y = att_weights.squeeze(3)[:, :, :T, :]  # (B, H, T, hs)

        # Reassemble all head outputs
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Apply output projection and dropout
        y = self.resid_dropout(self.c_proj(y))
        return y