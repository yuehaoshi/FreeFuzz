import numpy as np
import paddle
import paddle.nn as nn
import paddle.optimizer as opt

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4

IMAGE_SIZE = 784
CLASS_NUM = 10

# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1, )).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples

class LinearNet(nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = nn.Linear(IMAGE_SIZE, CLASS_NUM)
        self.bias = self._linear.bias

    @paddle.jit.to_static
    def forward(self, x):
        return self._linear(x)

def train(layer, loader, loss_fn, opt, model_average):
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            opt.step()
            model_average.step()
            opt.clear_grad()
            model_average.clear_grad()
            print("Train Epoch {} batch {}: loss = {}, bias = {}".format(
                epoch_id, batch_id, np.mean(loss.numpy()), layer.bias.numpy()))
def evaluate(layer, loader, loss_fn):
    for batch_id, (image, label) in enumerate(loader()):
        out = layer(image)
        loss = loss_fn(out, label)
        loss.backward()
        print("Evaluate batch {}: loss = {}, bias = {}".format(
            batch_id, np.mean(loss.numpy()), layer.bias.numpy()))

# create network
layer = LinearNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = opt.Momentum(learning_rate=0.2, momentum=0.1, parameters=layer.parameters())
model_average = paddle.incubate.ModelAverage(0.15,
                                            parameters=layer.parameters(),
                                            min_average_window=2,
                                            max_average_window=10)

# create data loader
dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=2)
# create data loader
eval_loader = paddle.io.DataLoader(dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=1)

# train
train(layer, loader, loss_fn, optimizer, model_average)

print("\nEvaluate With ModelAverage")
with model_average.apply(need_restore=False):
    evaluate(layer, eval_loader, loss_fn)

print("\nEvaluate With Restored Paramters")
model_average.restore()
evaluate(layer, eval_loader, loss_fn)