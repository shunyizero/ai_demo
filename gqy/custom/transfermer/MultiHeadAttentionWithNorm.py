import torch.nn as nn
import torch
from twisted.python.util import println

import MultiHeadAttention

# 编码器(transformer中的编码器)
# ================== 多头注意力机制 + 残差连接 + 层归一化 ==================
class MultiHeadAttentionWithNorm(nn.Module):
    def __init__(self, d_model, num_heads, dropout, d_input=None):
        super().__init__()
        self.mha = MultiHeadAttention.MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X_q, X_k, X_v):
        attn_output, attn_weights = self.mha(X_q, X_k, X_v)
        # 残差连接 + 层归一化
        out = self.norm(attn_output + X_q)
        return out, attn_weights

# ================== 测试方法 ==================
def test_multihead_attention_with_norm():
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2
    dropout = 0.1

    # 随机输入
    X_q = torch.randn(batch_size, seq_len, d_model)
    X_k = torch.randn(batch_size, seq_len, d_model)
    X_v = torch.randn(batch_size, seq_len, d_model)

    model = MultiHeadAttentionWithNorm(d_model, num_heads, dropout)
    out, attn_weights = model(X_q, X_k, X_v)
    println("输出形状:", out.shape)
    println("注意力权重形状:", attn_weights.shape)

if __name__ == "__main__":
    test_multihead_attention_with_norm()
