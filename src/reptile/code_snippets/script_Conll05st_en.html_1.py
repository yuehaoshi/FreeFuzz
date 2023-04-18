import paddle
from paddle.text.datasets import Conll05st

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, pred_idx, mark, label):
        return paddle.sum(pred_idx), paddle.sum(mark), paddle.sum(label)


conll05st = Conll05st()

for i in range(10):
    pred_idx, mark, label= conll05st[i][-3:]
    pred_idx = paddle.to_tensor(pred_idx)
    mark = paddle.to_tensor(mark)
    label = paddle.to_tensor(label)

    model = SimpleNet()
    pred_idx, mark, label= model(pred_idx, mark, label)
    print(pred_idx.numpy(), mark.numpy(), label.numpy())