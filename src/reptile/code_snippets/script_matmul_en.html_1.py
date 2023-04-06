import paddle
import numpy as np

# vector * vector
x_data = np.random.random([10]).astype(np.float32)
y_data = np.random.random([10]).astype(np.float32)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
z = paddle.matmul(x, y)
print(z.numpy().shape)
# [1]

# matrix * vector
x_data = np.random.random([10, 5]).astype(np.float32)
y_data = np.random.random([5]).astype(np.float32)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
z = paddle.matmul(x, y)
print(z.numpy().shape)
# [10]

# batched matrix * broadcasted vector
x_data = np.random.random([10, 5, 2]).astype(np.float32)
y_data = np.random.random([2]).astype(np.float32)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
z = paddle.matmul(x, y)
print(z.numpy().shape)
# [10, 5]

# batched matrix * batched matrix
x_data = np.random.random([10, 5, 2]).astype(np.float32)
y_data = np.random.random([10, 2, 5]).astype(np.float32)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
z = paddle.matmul(x, y)
print(z.numpy().shape)
# [10, 5, 5]

# batched matrix * broadcasted matrix
x_data = np.random.random([10, 1, 5, 2]).astype(np.float32)
y_data = np.random.random([1, 3, 2, 5]).astype(np.float32)
x = paddle.to_tensor(x_data)
y = paddle.to_tensor(y_data)
z = paddle.matmul(x, y)
print(z.numpy().shape)
# [10, 3, 5, 5]