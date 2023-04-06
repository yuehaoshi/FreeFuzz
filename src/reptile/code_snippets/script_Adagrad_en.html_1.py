import paddle
import numpy as np

inp = paddle.rand(shape=[10, 10])
linear = paddle.nn.Linear(10, 10)
out = linear(inp)
loss = paddle.mean(out)
adagrad = paddle.optimizer.Adagrad(learning_rate=0.1,
        parameters=linear.parameters())
out.backward()
adagrad.step()
adagrad.clear_grad()

#Note that the learning_rate of linear_2 is 0.01.
linear_1 = paddle.nn.Linear(10, 10)
linear_2 = paddle.nn.Linear(10, 10)
inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
out = linear_1(inp)
out = linear_2(out)
loss = paddle.mean(out)
adagrad = paddle.optimizer.Adagrad(
    learning_rate=0.1,
    parameters=[{
        'params': linear_1.parameters()
    }, {
        'params': linear_2.parameters(),
        'weight_decay': 0.001,
        'learning_rate': 0.1,
    }],
    weight_decay=0.01)
out.backward()
adagrad.step()
adagrad.clear_grad()