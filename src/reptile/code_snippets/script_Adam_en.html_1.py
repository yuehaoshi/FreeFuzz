import paddle

linear = paddle.nn.Linear(10, 10)
inp = paddle.rand([10,10], dtype="float32")
out = linear(inp)
loss = paddle.mean(out)
adam = paddle.optimizer.Adam(learning_rate=0.1,
        parameters=linear.parameters())
out.backward()
adam.step()
adam.clear_grad()