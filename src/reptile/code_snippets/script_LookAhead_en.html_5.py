import paddle
x = paddle.arange(26, dtype="float32").reshape([2, 13])

linear = paddle.nn.Linear(13, 5)
# This can be any optimizer supported by dygraph.
adam = paddle.optimizer.Adam(learning_rate = 0.01,
                            parameters = linear.parameters())
out = linear(x)
out.backward()
adam.step()
adam.clear_grad()