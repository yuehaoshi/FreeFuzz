import paddle

a = paddle.rand([2,13], dtype="float32")
linear = paddle.nn.Linear(13, 5)
# This can be any optimizer supported by dygraph.
opt = paddle.optimizer.AdamW(learning_rate = 0.01,
                            parameters = linear.parameters())
out = linear(a)
out.backward()
opt.step()
opt.clear_grad()