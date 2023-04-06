import paddle
from paddle.text.datasets import Movielens

class SimpleNet(paddle.nn.Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()

    def forward(self, category, title, rating):
        return paddle.sum(category), paddle.sum(title), paddle.sum(rating)


movielens = Movielens(mode='train')

for i in range(10):
    category, title, rating = movielens[i][-3:]
    category = paddle.to_tensor(category)
    title = paddle.to_tensor(title)
    rating = paddle.to_tensor(rating)

    model = SimpleNet()
    category, title, rating = model(category, title, rating)
    print(category.numpy().shape, title.numpy().shape, rating.numpy().shape)