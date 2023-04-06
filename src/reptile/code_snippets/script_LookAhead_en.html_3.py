import paddle
import numpy as np
inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
linear = paddle.nn.Linear(10, 1)
out = linear(inp)
loss = paddle.mean(out)
sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())
lookahead = paddle.incubate.LookAhead(sgd, alpha=0.2, k=5)
loss.backward()
lookahead.minimize(loss)
lookahead.clear_grad()