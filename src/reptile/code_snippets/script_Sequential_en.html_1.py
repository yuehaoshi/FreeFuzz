import paddle
import numpy as np

data = np.random.uniform(-1, 1, [30, 10]).astype('float32')
data = paddle.to_tensor(data)
# create Sequential with iterable Layers
model1 = paddle.nn.Sequential(
    paddle.nn.Linear(10, 1), paddle.nn.Linear(1, 2)
)
model1[0]  # access the first layer
res1 = model1(data)  # sequential execution

# create Sequential with name Layer pairs
model2 = paddle.nn.Sequential(
    ('l1', paddle.nn.Linear(10, 2)),
    ('l2', paddle.nn.Linear(2, 3))
)
model2['l1']  # access l1 layer
model2.add_sublayer('l3', paddle.nn.Linear(3, 3))  # add sublayer
res2 = model2(data)  # sequential execution