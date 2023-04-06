import paddle
import numpy as np

x = np.random.random(size=(2, 3, 4, 5)).astype('float32')
x = paddle.to_tensor(x)
m = paddle.nn.Dropout2D(p=0.5)
y_train = m(x)
m.eval()  # switch the model to test phase
y_test = m(x)
print(x)
print(y_train)
print(y_test)