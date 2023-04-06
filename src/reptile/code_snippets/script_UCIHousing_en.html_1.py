import paddle
from paddle.text.datasets import UCIHousing

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, feature, target):
        return paddle.sum(feature), target

paddle.disable_static()

uci_housing = UCIHousing(mode='train')

for i in range(10):
    feature, target = uci_housing[i]
    feature = paddle.to_tensor(feature)
    target = paddle.to_tensor(target)

    model = SimpleNet()
    feature, target = model(feature, target)
    print(feature.numpy().shape, target.numpy())