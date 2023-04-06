import paddle

inp = paddle.uniform(shape=[10, 10], dtype='float32', min=-0.1, max=0.1)
linear = paddle.nn.Linear(10, 10)
out = linear(inp)
loss = paddle.mean(out)
beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.85], dtype="float32")
lamb = paddle.optimizer.Lamb(learning_rate=0.002, parameters=linear.parameters(), lamb_weight_decay=0.01)
back = out.backward()
lamb.step()
lamb.clear_grad()