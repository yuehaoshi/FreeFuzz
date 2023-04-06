# Adam with beta1/beta2 as Tensor and weight_decay as float
import paddle

linear = paddle.nn.Linear(10, 10)
inp = paddle.rand([10,10], dtype="float32")
out = linear(inp)
loss = paddle.mean(out)

beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.99], dtype="float32")

adam = paddle.optimizer.Adam(learning_rate=0.1,
        parameters=linear.parameters(),
        beta1=beta1,
        beta2=beta2,
        weight_decay=0.01)
out.backward()
adam.step()
adam.clear_grad()

#Note that the learning_rate of linear_2 is 0.01.
linear_1 = paddle.nn.Linear(10, 10)
linear_2 = paddle.nn.Linear(10, 10)
inp = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
out = linear_1(inp)
out = linear_2(out)
loss = paddle.mean(out)
adam = paddle.optimizer.Adam(
    learning_rate=0.1,
    parameters=[{
        'params': linear_1.parameters()
    }, {
        'params': linear_2.parameters(),
        'weight_decay': 0.001,
        'learning_rate': 0.1,
        'beta1': 0.8
    }],
    weight_decay=0.01,
    beta1=0.9)
out.backward()
adam.step()
adam.clear_grad()