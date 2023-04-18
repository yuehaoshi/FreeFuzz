import paddle

m = paddle.nn.LeakyReLU()
x = paddle.to_tensor([-2.0, 0, 1])
out = m(x)  # [-0.02, 0., 1.]