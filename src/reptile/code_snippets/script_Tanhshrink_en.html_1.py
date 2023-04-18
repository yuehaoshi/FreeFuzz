import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3])
m = paddle.nn.Tanhshrink()
out = m(x)
print(out)
# Tensor(shape=[4], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [-0.02005106, -0.00262468,  0.00033200,  0.00868741])