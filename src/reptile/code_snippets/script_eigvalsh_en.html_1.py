import numpy as np
import paddle

x_data = np.array([[1, -2j], [2j, 5]])
x = paddle.to_tensor(x_data)
out_value = paddle.eigvalsh(x, UPLO='L')
print(out_value)
#[0.17157288, 5.82842712]