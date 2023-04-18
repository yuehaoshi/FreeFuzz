import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
from paddle.static import InputSpec

class Mnist(nn.Layer):
    def __init__(self):
        super(Mnist, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 200),
            nn.Tanh(),
            nn.Linear(200, 10),
            nn.Softmax())

    def forward(self, x):
        return self.net(x)

dynamic = True  # False
# if use static graph, do not set
if not dynamic:
    paddle.enable_static()

input = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')
model = paddle.Model(Mnist(), input, label)
optim = paddle.optimizer.SGD(learning_rate=1e-3,
    parameters=model.parameters())
model.prepare(optim, paddle.nn.CrossEntropyLoss())

transform = T.Compose([
    T.Transpose(),
    T.Normalize([127.5], [127.5])
])
data = paddle.vision.datasets.MNIST(mode='train', transform=transform)

model.fit(data, epochs=1, batch_size=32, verbose=0)
model.save('checkpoint/test')  # save for training
model.save('inference_model', False)  # save for inference