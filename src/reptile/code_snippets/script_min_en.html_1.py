import paddle

# x is a tensor with shape [2, 4]
# the axis is a int element
x = paddle.to_tensor([[0.2, 0.3, 0.5, 0.9],
                      [0.1, 0.2, 0.6, 0.7]])
result1 = paddle.min(x)
print(result1)
#[0.1]
result2 = paddle.min(x, axis=0)
print(result2)
#[0.1 0.2 0.5 0.7]
result3 = paddle.min(x, axis=-1)
print(result3)
#[0.2 0.1]
result4 = paddle.min(x, axis=1, keepdim=True)
print(result4)
#[[0.2]
# [0.1]]

# y is a Tensor with shape [2, 2, 2]
# the axis is list
y = paddle.to_tensor([[[1.0, 2.0], [3.0, 4.0]],
                      [[5.0, 6.0], [7.0, 8.0]]])
result5 = paddle.min(y, axis=[1, 2])
print(result5)
#[1. 5.]
result6 = paddle.min(y, axis=[0, 1])
print(result6)
#[1. 2.]