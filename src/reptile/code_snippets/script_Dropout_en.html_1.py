import paddle
import numpy as np

x = np.array([[1,2,3], [4,5,6]]).astype('float32')
x = paddle.to_tensor(x)
m = paddle.nn.Dropout(p=0.5)
y_train = m(x)
m.eval()  # switch the model to test phase
y_test = m(x)
print(x)
print(y_train)
print(y_test)