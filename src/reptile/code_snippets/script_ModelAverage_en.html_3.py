import paddle
inp = paddle.rand([1, 10], dtype="float32")
linear = paddle.nn.Linear(10, 1)
out = linear(inp)
loss = paddle.mean(out)
loss.backward()

sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
sgd.minimize(loss)

modelaverage = paddle.incubate.ModelAverage(0.15,
                                            parameters=linear.parameters(),
                                            min_average_window=2,
                                            max_average_window=4)
modelaverage.minimize(loss)
sgd.clear_grad()
modelaverage.clear_grad()