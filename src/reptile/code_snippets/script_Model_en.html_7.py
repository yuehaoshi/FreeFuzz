import paddle
import paddle.nn as nn
from paddle.static import InputSpec

device = paddle.set_device('cpu')

input = InputSpec([None, 784], 'float32', 'x')

model = paddle.Model(nn.Sequential(
    nn.Linear(784, 200),
    nn.Tanh(),
    nn.Linear(200, 10),
    nn.Softmax()), input)

model.save('checkpoint/test')
model.load('checkpoint/test')