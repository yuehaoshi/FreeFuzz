import paddle

x = paddle.rand((2, 2, 2, 3))
layer_norm = paddle.nn.LayerNorm(x.shape[1:])
layer_norm_out = layer_norm(x)

print(layer_norm_out)