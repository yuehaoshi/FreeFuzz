import paddle

inp = paddle.uniform([10, 10], dtype="float32", min=-0.1, max=0.1)
linear = paddle.nn.Linear(10, 10)
out = linear(inp)
loss = paddle.mean(out)
optimizer = paddle.optimizer.Adam(learning_rate=0.1,
        parameters=linear.parameters())
params_grads = optimizer.backward(loss)
optimizer.apply_gradients(params_grads)