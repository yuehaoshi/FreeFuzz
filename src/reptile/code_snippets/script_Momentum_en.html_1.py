import paddle
import numpy as np
inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
linear = paddle.nn.Linear(10, 10)
inp = paddle.to_tensor(inp)
out = linear(inp)
loss = paddle.mean(out)
beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.99], dtype="float32")
momentum = paddle.optimizer.Momentum(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
back = out.backward()
momentum.step()
momentum.clear_grad()

#Note that the learning_rate of linear_2 is 0.01.
linear_1 = paddle.nn.Linear(10, 10)
linear_2 = paddle.nn.Linear(10, 10)
inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
out = linear_1(inp)
out = linear_2(out)
loss = paddle.mean(out)
momentum = paddle.optimizer.Momentum(
    learning_rate=0.1,
    parameters=[{
        'params': linear_1.parameters()
    }, {
        'params': linear_2.parameters(),
        'weight_decay': 0.001,
        'learning_rate': 0.1
    }],
    weight_decay=0.01,
    momentum=0.9)
out.backward()
momentum.step()
momentum.clear_grad()