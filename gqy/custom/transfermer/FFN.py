import torch.nn as nn
import torch

"""
    FNN: 前馈神经网络(用于transformer-attention后的前馈处理)
"""
class FFN(nn.Module):
    def __init__(self, d_model, hidden_dim_multiplier=4, resid_pdrop=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, hidden_dim_multiplier * d_model)
        self.act = nn.ReLU(True)  # inplace=True, saves a little bit of memory
        self.proj = nn.Linear(hidden_dim_multiplier * d_model, d_model)
        self.dropout = nn.Dropout(resid_pdrop)

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        x = self.dropout(self.proj(self.act(self.fc_1(x))))  #  [batch_size, seq_len, embed_dim]
        return x

if __name__ == "__main__":
    # 测试FFN
    batch_size = 2
    seq_len = 4
    d_model = 8

    # 随机输入
    x = torch.randn(batch_size, seq_len, d_model)

    model = FFN(d_model)
    out = model(x)
    print("输出形状:", out.shape)  # 应该是 [batch_size, seq_len, d_model]