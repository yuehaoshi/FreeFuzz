import paddle

x = paddle.to_tensor([-2., 0., 1.])
m = paddle.nn.ReLU()
out = m(x) # [0., 0., 1.]