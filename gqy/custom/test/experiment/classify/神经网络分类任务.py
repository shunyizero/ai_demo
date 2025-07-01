from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np

#  获取目录s
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"
# 构建目录
PATH.mkdir(parents=True, exist_ok=True)

# 注意pkl版本跟Yann LeCun官方版的不同
URL = "http://www.iro.umontreal.ca/~lisa/deep/data/mnist/"
FILENAME = "mnist.pkl.gz"
if not (PATH / FILENAME).exists():
    response = requests.get(URL + FILENAME)
    # 如果获取不到内容，则抛出异常
    if response.status_code != 200:
        raise Exception(f"Failed to download {FILENAME} from {URL}, status code: {response.status_code}")
    (PATH / FILENAME).open("wb").write(response.content)

# 读取数据
with gzip.open(PATH / FILENAME, "rb") as f:
    print(type(f) )
    ((x_train,y_train),(x_valid,y_valid),_) = pickle.load(f, encoding="latin-1")
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# 绘图
# image0 = x_train[0].reshape(28,28)
# pyplot.imshow(image0, cmap="gray")
# pyplot.show()
#
# image1 = x_train[1].reshape(28,28)
# pyplot.imshow(image1, cmap="gray")
# pyplot.show()

# 转成Tensor
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
print(f"y_train: {y_train.max()}, {y_train.min()}, {y_train.unique()}")

# 交叉熵损失函数
loss_func = F.cross_entropy
# y = wx+b
def model(xb):
    # 模型
    return xb.mm(weight) + bias

# 从训练集中抽取批次数据
bs = 64
xb = x_train[0:bs] # 64, 784
yb = y_train[0:bs]

weight = torch.randn([784, 10], dtype=torch.float, requires_grad=True)
bias = torch.zeros(10, requires_grad=True)
print(loss_func(model(xb), yb))

# 定义模型
class Minst_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(784, 128)
        self.hidden2 = nn.Linear(128, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

net = Minst_NN()
print(net)

#  打印模型参数
for name, param in net.named_parameters():
    print(name, param, param.size())

train_ds = TensorDataset(x_train, y_train)
# train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
valid_ds = TensorDataset(x_valid, y_valid)
# valid_dl = DataLoader(valid_ds, batch_size=bs*2, shuffle=True)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs*2)
    )

# 训练
# def fit(steps, model, loss_func, opt, train_dl, valid_dl):
#     for step in range(steps):
#         model.train()
#         # 训练
#         for xb, yb in train_dl:
#             loss_batch(model, loss_func, xb, yb, opt)
#
#         model.eval()
#         # 验证
#         with torch.no_grad():
#             losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
#         val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
#         print(f"step: {step}, val_loss: {val_loss}")

def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums, corrects = [], [], []
            for xb, yb in valid_dl:
                pred = model(xb)
                loss = loss_func(pred, yb)
                losses.append(loss.item())
                nums.append(len(xb))
                correct = (pred.argmax(dim=1) == yb).sum().item()
                corrects.append(correct)
            val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
            accuracy = np.sum(corrects) / np.sum(nums)
        print(f"step: {step}, val_loss: {val_loss:.4f}, val_acc: {accuracy:.4f}")

# 优化器
def loss_batch(model, loss_func, xb, yb, opt=None):
    # 计算损失
    pred = model(xb)
    loss = loss_func(pred, yb)
    if opt is not None:
        # 反向传播
        loss.backward()
        # 更新参数
        opt.step()
        # 梯度清零
        opt.zero_grad()
    return loss.item(), len(xb)

def get_model():
    model = Minst_NN()
    return model, optim.AdamW(model.parameters(), lr=0.0001)

train_dl, valid_dl = get_data(train_ds,valid_ds,bs)
model,opt = get_model()
fit(50, model, loss_func, opt, train_dl, valid_dl)



