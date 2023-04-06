import paddle

rnn = paddle.nn.SimpleRNN(16, 32, 2)

x = paddle.randn((4, 23, 16))
prev_h = paddle.randn((2, 4, 32))
y, h = rnn(x, prev_h)

print(y.shape)
print(h.shape)

#[4,23,32]
#[2,4,32]