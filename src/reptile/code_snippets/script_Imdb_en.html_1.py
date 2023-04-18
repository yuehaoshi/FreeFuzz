import paddle
from paddle.text.datasets import Imdb

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, doc, label):
        return paddle.sum(doc), label


imdb = Imdb(mode='train')

for i in range(10):
    doc, label = imdb[i]
    doc = paddle.to_tensor(doc)
    label = paddle.to_tensor(label)

    model = SimpleNet()
    image, label = model(doc, label)
    print(doc.numpy().shape, label.numpy().shape)