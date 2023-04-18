import paddle
from paddle.text.datasets import Imikolov

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, src, trg):
        return paddle.sum(src), paddle.sum(trg)


imikolov = Imikolov(mode='train', data_type='SEQ', window_size=2)

for i in range(10):
    src, trg = imikolov[i]
    src = paddle.to_tensor(src)
    trg = paddle.to_tensor(trg)

    model = SimpleNet()
    src, trg = model(src, trg)
    print(src.numpy().shape, trg.numpy().shape)