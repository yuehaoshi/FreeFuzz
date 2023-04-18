import paddle
import numpy as np

value = np.arange(26).reshape(2, 13).astype("float32")
a = paddle.to_tensor(value)
linear = paddle.nn.Linear(13, 5)
adam = paddle.optimizer.Adam(learning_rate=0.01,
                            parameters=linear.parameters())
out = linear(a)
out.backward()
adam.step()
linear.clear_gradients()