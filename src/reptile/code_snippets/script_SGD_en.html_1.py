import paddle

inp = paddle.uniform(min=-0.1, max=0.1, shape=[10, 10], dtype='float32')
linear = paddle.nn.Linear(10, 10)
inp = paddle.to_tensor(inp)
out = linear(inp)
loss = paddle.mean(out)
sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
out.backward()
sgd.step()
sgd.clear_grad()