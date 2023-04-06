import numpy as np
import paddle
import paddle.nn as nn
from paddle.static import InputSpec

device = paddle.set_device('cpu') # or 'gpu'

net = nn.Sequential(
    nn.Linear(784, 200),
    nn.Tanh(),
    nn.Linear(200, 10))

input = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')
model = paddle.Model(net, input, label)
optim = paddle.optimizer.SGD(learning_rate=1e-3,
    parameters=model.parameters())
model.prepare(optim, paddle.nn.CrossEntropyLoss())
data = np.random.random(size=(4,784)).astype(np.float32)
label = np.random.randint(0, 10, size=(4, 1)).astype(np.int64)
loss = model.train_batch([data], [label])
print(loss)