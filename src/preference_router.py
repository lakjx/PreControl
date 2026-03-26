import torch
import torch.nn as nn
import math

class PreferenceRouter(nn.Module):
    def __init__(self, hidden_dim: int, rank: int, num_attributes: int, lora_alpha: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.num_attributes = num_attributes
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        
        # A matrix: down-projection (features become rank)
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        # B matrix: up-projection (rank becomes hidden_dim)
        self.B = nn.Linear(rank, hidden_dim, bias=False)
        
        # W matrices: K attribute routers of size [rank, rank]
        self.W = nn.Parameter(torch.empty(num_attributes, rank, rank))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Init A like LoRA (Kaiming uniform)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        # Init B as zero so initial editing is identity (like LoRA)
        nn.init.zeros_(self.B.weight)
        # Init W as ZEROS so router starts from no-edit state
        # This ensures ortho loss starts at 0 and grows naturally
        nn.init.zeros_(self.W)

    def forward(self, h: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """
        h: [batch_size, seq_len, hidden_dim]
        alphas: [batch_size, num_attributes]
        """
        # 1. Project to rank
        h_proj = self.A(h) # [batch_size, seq_len, rank]
        
        # 2. Combine W based on alphas
        W_batch = torch.einsum('bk,krs->brs', alphas, self.W)
        
        # 3. Apply W_batch to h_proj
        # h_proj: [b, s, r_in]
        # W_batch: [b, r_out, r_in] ... wait, W is [rank, rank]
        # If we treat h_proj as [b, s, r_in], we want h_proj @ W_batch^T
        # h_proj[b, s, i] * W_batch[b, r, i] -> [b, s, r]
        # So: einsum('bsi,bri->bsr', h_proj, W_batch)
        # Let's ensure W_batch shape interpretation.
        # W_batch is [b, r_out, r_in]
        # Output h_routed = torch.bmm(h_proj, W_batch.transpose(1, 2))
        h_routed = torch.einsum('bsi,bri->bsr', h_proj, W_batch)
        
        # 4. Project back to hidden_dim
        delta_h = self.B(h_routed) # [batch_size, seq_len, hidden_dim]
        
        # 5. Add to original
        h_edited = h + self.scaling * delta_h
        return h_edited

    def get_ortho_loss(self) -> torch.Tensor:
        """
        Computes the orthogonal loss between different attribute routers W_i.
        L_ortho = sum_{i != j} || W_i W_j^T ||_F^2
        """
        # W: [K, r, r]
        # interaction[i, j, a, c] = sum_b W[i, a, b] * W[j, c, b]
        interaction = torch.einsum('iab,jcb->ijac', self.W, self.W)
        
        # Frobenius norm squared
        norms = torch.sum(interaction ** 2, dim=(2, 3)) # [K, K]
        
        # Mask out the diagonal (i == j)
        K = self.num_attributes
        mask = 1 - torch.eye(K, device=self.W.device, dtype=norms.dtype)
        
        ortho_loss = torch.sum(norms * mask)
        return ortho_loss
