import paddle

x = paddle.to_tensor([-0.4, -0.2, 0.1, 0.3], dtype='float32')
m = paddle.nn.Softplus()
out = m(x) # [0.513015, 0.598139, 0.744397, 0.854355]