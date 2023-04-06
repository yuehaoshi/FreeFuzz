import paddle
linear = paddle.nn.Linear(10, 10)
input = paddle.uniform(shape=[10, 10], min=-0.1, max=0.1)
out = linear(input)
loss = paddle.mean(out)

beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.99], dtype="float32")

adam = paddle.optimizer.Adam(learning_rate=0.1,
        parameters=linear.parameters(),
        weight_decay=0.01)
out.backward()
adam.minimize(loss)
adam.clear_grad()