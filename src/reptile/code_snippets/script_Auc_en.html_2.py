import numpy as np
import paddle
import paddle.nn as nn

class Data(paddle.io.Dataset):
    def __init__(self):
        super(Data, self).__init__()
        self.n = 1024
        self.x = np.random.randn(self.n, 10).astype('float32')
        self.y = np.random.randint(2, size=(self.n, 1)).astype('int64')

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.n

model = paddle.Model(nn.Sequential(
    nn.Linear(10, 2), nn.Softmax())
)
optim = paddle.optimizer.Adam(
    learning_rate=0.001, parameters=model.parameters())

def loss(x, y):
    return nn.functional.nll_loss(paddle.log(x), y)

model.prepare(
    optim,
    loss=loss,
    metrics=paddle.metric.Auc())
data = Data()
model.fit(data, batch_size=16)