import torch.nn as nn
import torch
import numpy as np

class MultiHeadAttention(nn.Module):

    # 多头注意力申请网络定义(初始化)
    def __init__(self, d_model, num_heads, dropout, d_input=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        if d_input is None:
            d_xq = d_xk = d_xv = d_model
        else:
            d_xq, d_xk, d_xv = d_input

        # Make sure that the embedding dimension of model is a multiple of number of heads
        assert d_model % self.num_heads == 0

        self.d_k = d_model // self.num_heads  # here d is divided between the heads
        # each head has hidden dimension d

        # These are still of dimension d_model. They will be split into number of heads
        self.W_q = nn.Linear(d_xq, d_model, bias=False)
        self.W_k = nn.Linear(d_xk, d_model, bias=False)
        self.W_v = nn.Linear(d_xv, d_model, bias=False)

        # Outputs of all sub-layers need to be of dimension d_model
        self.W_h = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    # 自注意机制公式计算(attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V)
    def scaled_dot_product_attention(self, Q, K, V):
        batch_size = Q.size(0)
        k_length = K.size(-2)

        # Scaling by d_k so that the soft(arg)max doesn't saturate
        Q = Q / np.sqrt(self.d_k)                         # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(Q, K.transpose(2, 3))          # (bs, n_heads, q_length, k_length)

        A = torch.softmax(scores, dim=-1)  # (bs, n_heads, q_length, k_length)
        A = self.dropout(A)

        # Get the weighted average of the values
        H = torch.matmul(A, V)     # (bs, n_heads, q_length, dim_per_head)

        return H, A

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (heads X depth)
        Return after transpose to put in shape (batch_size X num_heads X seq_length X d_k)
        """
        return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

    def group_heads(self, x, batch_size):
        """
        Combine the heads again to get (batch_size X seq_length X (num_heads times d_k))
        """
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)

    # 定义前向传播函数
    def forward(self, X_q, X_k, X_v):
        batch_size, seq_length, dim = X_q.size() # dim = embedding dimension

        # After transforming, split into num_heads
        Q = self.split_heads(self.W_q(X_q), batch_size)  # (bs, n_heads, q_length, dim_per_head)
        K = self.split_heads(self.W_k(X_k), batch_size)  # (bs, n_heads, k_length, dim_per_head)
        V = self.split_heads(self.W_v(X_v), batch_size)  # (bs, n_heads, v_length, dim_per_head)

        # Calculate the attention weights for each of the heads
        H_cat, A = self.scaled_dot_product_attention(Q, K, V)

        # Put all the heads back together by concat
        H_cat = self.group_heads(H_cat, batch_size)    # (bs, q_length, dim)

        # Final linear layer
        H = self.W_h(H_cat)  # (bs, q_length, dim)

        return H, A

# 测试(忽略)
if __name__ == "__main__":
    # 参数设置
    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2
    dropout = 0.1

    # 随机输入
    X_q = torch.randn(batch_size, seq_length, d_model)
    X_k = torch.randn(batch_size, seq_length, d_model)
    X_v = torch.randn(batch_size, seq_length, d_model)

    # 实例化模型
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

    # 前向传播
    output, attn = mha(X_q, X_k, X_v)

    print("输出 shape:", output.shape)  # 期望: (batch_size, seq_length, d_model)
    print("注意力 shape:", attn.shape)  # 期望: (batch_size, num_heads, seq_length, seq_length)
