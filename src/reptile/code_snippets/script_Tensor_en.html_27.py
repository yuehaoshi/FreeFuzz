import paddle
import numpy as np

a = np.random.rand(3, 3)
a_t = np.transpose(a, [1, 0])
x_data = np.matmul(a, a_t) + 1e-03
x = paddle.to_tensor(x_data)
out = paddle.cholesky(x, upper=False)
print(out)
# [[1.190523   0.         0.        ]
#  [0.9906703  0.27676893 0.        ]
#  [1.25450498 0.05600871 0.06400121]]