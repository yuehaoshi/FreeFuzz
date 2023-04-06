import numpy as np
import paddle
from paddle.static import InputSpec

class MnistDataset(paddle.vision.datasets.MNIST):
    def __init__(self, mode, return_label=True):
        super(MnistDataset, self).__init__(mode=mode)
        self.return_label = return_label

    def __getitem__(self, idx):
        img = np.reshape(self.images[idx], [1, 28, 28])
        if self.return_label:
            return img, np.array(self.labels[idx]).astype('int64')
        return img,

    def __len__(self):
        return len(self.images)

test_dataset = MnistDataset(mode='test', return_label=False)

# imperative mode
input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
model = paddle.Model(paddle.vision.models.LeNet(), input)
model.prepare()
result = model.predict(test_dataset, batch_size=64)
print(len(result[0]), result[0][0].shape)

# declarative mode
device = paddle.set_device('cpu')
paddle.enable_static()
input = InputSpec([-1, 1, 28, 28], 'float32', 'image')
model = paddle.Model(paddle.vision.models.LeNet(), input)
model.prepare()

result = model.predict(test_dataset, batch_size=64)
print(len(result[0]), result[0][0].shape)