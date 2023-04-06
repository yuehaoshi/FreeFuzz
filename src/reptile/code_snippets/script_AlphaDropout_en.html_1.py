import paddle
import numpy as np

x = np.array([[-1, 1], [-1, 1]]).astype('float32')
x = paddle.to_tensor(x)
m = paddle.nn.AlphaDropout(p=0.5)
y_train = m(x)
m.eval()  # switch the model to test phase
y_test = m(x)
print(x)
print(y_train)
# [[-0.10721093, 1.6655989 ], [-0.7791938, -0.7791938]] (randomly)
print(y_test)