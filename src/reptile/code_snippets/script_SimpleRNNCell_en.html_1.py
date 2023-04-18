import paddle

x = paddle.randn((4, 16))
prev_h = paddle.randn((4, 32))

cell = paddle.nn.SimpleRNNCell(16, 32)
y, h = cell(x, prev_h)
print(y.shape)

#[4,32]