#Take the subclass adam as an example
import paddle
linear = paddle.nn.Linear(10, 10)
inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
out = linear(inp)
loss = paddle.mean(out)
adam = paddle.optimizer.Adam(learning_rate=0.1,
        parameters=linear.parameters())
out.backward()
adam.step()
adam.clear_grad()

#Take the subclass sgd as an example
#optimize parameters in linear_1 and linear2 in different options.
#Note that the learning_rate of linear_2 is 0.01.
linear_1 = paddle.nn.Linear(10, 10)
linear_2 = paddle.nn.Linear(10, 10)
inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
out = linear_1(inp)
out = linear_2(out)
loss = paddle.mean(out)
sgd = paddle.optimizer.SGD(
    learning_rate=0.1,
    parameters=[{
        'params': linear_1.parameters()
    }, {
        'params': linear_2.parameters(),
        'weight_decay': 0.001,
        'learning_rate': 0.1
    }],
    weight_decay=0.01)
out.backward()
sgd.step()
sgd.clear_grad()