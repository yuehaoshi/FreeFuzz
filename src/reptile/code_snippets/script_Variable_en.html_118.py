import paddle
import numpy as np

# A * B
A_data = np.random.random([3, 4]).astype(np.float32)
B_data = np.random.random([4, 5]).astype(np.float32)
A = paddle.to_tensor(A_data)
B = paddle.to_tensor(B_data)
out = paddle.linalg.multi_dot([A, B])
print(out.numpy().shape)
# [3, 5]

# A * B * C
A_data = np.random.random([10, 5]).astype(np.float32)
B_data = np.random.random([5, 8]).astype(np.float32)
C_data = np.random.random([8, 7]).astype(np.float32)
A = paddle.to_tensor(A_data)
B = paddle.to_tensor(B_data)
C = paddle.to_tensor(C_data)
out = paddle.linalg.multi_dot([A, B, C])
print(out.numpy().shape)
# [10, 7]