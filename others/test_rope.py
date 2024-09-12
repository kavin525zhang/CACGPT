import torch
import math

def rotary_position_embedding(q, k):
    """
    Rotary Position Embedding (RoPE) for queries and keys.
    
    Args:
        q: tensor for queries of shape (batch_size, num_heads, seq_len, dim)
        k: tensor for keys of shape (batch_size, num_heads, seq_len, dim)
        
    Returns:
        Rotated queries and keys
    """
    batch_size, num_heads, seq_len, dim = q.size()
    
    # Begin of sinusoidal_position_embedding content
    # 序列对应的位置序号
    position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1).to(q.device)
    # q维度上的theta值
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim)).to(q.device)
    
    pos_emb = position * div_term
    pos_emb = torch.stack([torch.sin(pos_emb), torch.cos(pos_emb)], dim=-1).flatten(-2, -1)
    pos_emb = pos_emb.unsqueeze(0).unsqueeze(1)
    pos_emb = pos_emb.expand(batch_size, num_heads, -1, -1)
    # End of sinusoidal_position_embedding content

    # Extract and duplicate cosine and sine embeddings
    cos_emb = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
    sin_emb = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # Create alternate versions of q and k
    q_alternate = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.size())
    k_alternate = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.size())

    # Rotate queries and keys
    q_rotated = q * cos_emb + q_alternate * sin_emb
    k_rotated = k * cos_emb + k_alternate * sin_emb

    return q_rotated, k_rotated