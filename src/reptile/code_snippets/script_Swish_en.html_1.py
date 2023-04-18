import paddle

x = paddle.to_tensor([-2., 0., 1.])
m = paddle.nn.Swish()
out = m(x)
print(out)
# Tensor(shape=[3], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-0.23840584,  0.        ,  0.73105854])