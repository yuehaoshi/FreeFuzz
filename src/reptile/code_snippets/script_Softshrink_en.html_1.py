import paddle

x = paddle.to_tensor([-0.9, -0.2, 0.1, 0.8])
m = paddle.nn.Softshrink()
out = m(x)
print(out)
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-0.39999998,  0.        ,  0.        ,  0.30000001])