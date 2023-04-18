import paddle

x = paddle.to_tensor(5., stop_gradient=False)
y = paddle.pow(x, 4.0)
y.backward()
print("grad of x: {}".format(x.gradient()))
# [500.]