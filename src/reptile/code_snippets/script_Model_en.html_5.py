import paddle
import paddle.nn as nn
from paddle.static import InputSpec

device = paddle.set_device('cpu') # or 'gpu'

input = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')

net = nn.Sequential(
    nn.Linear(784, 200),
    nn.Tanh(),
    nn.Linear(200, 10),
    nn.Softmax())

model = paddle.Model(net, input, label)
model.prepare()
data = paddle.rand((1, 784), dtype="float32")
out = model.predict_batch([data])
print(out)
# [array([[0.08189095, 0.16740078, 0.06889386, 0.05085445, 0.10729759,
#          0.02217775, 0.14518553, 0.1591538 , 0.01808308, 0.17906217]],
#          dtype=float32)]