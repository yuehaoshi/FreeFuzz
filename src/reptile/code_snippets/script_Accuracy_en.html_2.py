import paddle
from paddle.static import InputSpec
import paddle.vision.transforms as T
from paddle.vision.datasets import MNIST

input = InputSpec([None, 1, 28, 28], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')
transform = T.Compose([T.Transpose(), T.Normalize([127.5], [127.5])])
train_dataset = MNIST(mode='train', transform=transform)

model = paddle.Model(paddle.vision.models.LeNet(), input, label)
optim = paddle.optimizer.Adam(
    learning_rate=0.001, parameters=model.parameters())
model.prepare(
    optim,
    loss=paddle.nn.CrossEntropyLoss(),
    metrics=paddle.metric.Accuracy())

model.fit(train_dataset, batch_size=64)