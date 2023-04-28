import paddle
import paddle.vision.transforms as T
from paddle.vision.datasets import MNIST
from paddle.static import InputSpec

dynamic = True
if not dynamic:
  paddle.enable_static()

transform = T.Compose([
  T.Transpose(),
  T.Normalize([127.5], [127.5])
])
train_dataset = MNIST(mode='train', transform=transform)
val_dataset = MNIST(mode='test', transform=transform)

input = InputSpec([None, 1, 28, 28], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')

model = paddle.Model(
  paddle.vision.models.LeNet(),
  input, label)
optim = paddle.optimizer.Adam(
  learning_rate=0.001, parameters=model.parameters())
model.prepare(
  optim,
  paddle.nn.CrossEntropyLoss(),
  paddle.metric.Accuracy(topk=(1, 2)))
model.fit(train_dataset,
          val_dataset,
          epochs=2,
          batch_size=64,
          save_dir='mnist_checkpoint')