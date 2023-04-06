import paddle
import paddle.vision.transforms as T
from paddle.static import InputSpec

# declarative mode
transform = T.Compose([
        T.Transpose(),
        T.Normalize([127.5], [127.5])
    ])
val_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)

input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')
model = paddle.Model(paddle.vision.models.LeNet(), input, label)
model.prepare(metrics=paddle.metric.Accuracy())
result = model.evaluate(val_dataset, batch_size=64)
print(result)