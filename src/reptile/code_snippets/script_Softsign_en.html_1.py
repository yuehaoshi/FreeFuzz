import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
m = paddle.nn.Softsign()
out = m(x)
print(out)
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-0.28571430, -0.16666666,  0.09090909,  0.23076925])