import paddle

x = paddle.to_tensor([-1., 0.3, 6.5])
m = paddle.nn.ReLU6()
out = m(x)
print(out)
# [0, 0.3, 6]