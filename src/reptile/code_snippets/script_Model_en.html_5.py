import numpy as np
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
data = np.random.random(size=(4,784)).astype(np.float32)
out = model.predict_batch([data])
print(out)