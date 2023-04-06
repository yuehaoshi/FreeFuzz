import paddle
import numpy as np
inp = paddle.to_tensor(np.random.random([1, 10]).astype('float32'))
linear = paddle.nn.Linear(10, 1)
out = linear(inp)
loss = paddle.mean(out)
loss.backward()

sgd = paddle.optimizer.SGD(learning_rate=0.1,parameters=linear.parameters())

modelaverage = paddle.incubate.ModelAverage(0.15,
                                            parameters=linear.parameters(),
                                            min_average_window=2,
                                            max_average_window=4)
sgd.step()
modelaverage.step()

with modelaverage.apply():
    for param in linear.parameters():
        print(param)

for param in linear.parameters():
    print(param)