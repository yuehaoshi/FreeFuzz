import paddle

a = paddle.rand([2,13], dtype="float32")
linear = paddle.nn.Linear(13, 5)
# This can be any optimizer supported by dygraph.
adam = paddle.optimizer.Adam(learning_rate = 0.01,
                            parameters = linear.parameters())
out = linear(a)
out.backward()
adam.step()
adam.clear_grad()