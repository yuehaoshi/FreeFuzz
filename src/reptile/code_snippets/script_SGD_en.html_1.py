import paddle
import numpy as np
inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
linear = paddle.nn.Linear(10, 10)
inp = paddle.to_tensor(inp)
out = linear(inp)
loss = paddle.mean(out)
beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.99], dtype="float32")
sgd = paddle.optimizer.SGD(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
back = out.backward()
sgd.step()
sgd.clear_grad()