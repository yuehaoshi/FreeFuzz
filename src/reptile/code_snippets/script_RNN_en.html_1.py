import paddle

inputs = paddle.rand((4, 23, 16))
prev_h = paddle.randn((4, 32))

cell = paddle.nn.SimpleRNNCell(16, 32)
rnn = paddle.nn.RNN(cell)
outputs, final_states = rnn(inputs, prev_h)

print(outputs.shape)
print(final_states.shape)

#[4,23,32]
#[4,32]