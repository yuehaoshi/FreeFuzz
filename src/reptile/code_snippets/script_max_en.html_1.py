import paddle

# data_x is a Tensor with shape [2, 4]
# the axis is a int element

x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]])
result1 = paddle.max(x)
print(result1)
#[0.9]
result2 = paddle.max(x, axis=0)
print(result2)
#[0.2 0.3 0.6 0.9]
result3 = paddle.max(x, axis=-1)
print(result3)
#[0.9 0.7]
result4 = paddle.max(x, axis=1, keepdim=True)
print(result4)
#[[0.9]
# [0.7]]

# data_y is a Tensor with shape [2, 2, 2]
# the axis is list

y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]])
result5 = paddle.max(y, axis=[1, 2])
print(result5)
#[4. 8.]
result6 = paddle.max(y, axis=[0, 1])
print(result6)
#[7. 8.]