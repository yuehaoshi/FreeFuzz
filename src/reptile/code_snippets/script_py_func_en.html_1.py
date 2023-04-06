# example 1:
import paddle
import six
import numpy as np

paddle.enable_static()

# Creates a forward function, Tensor can be input directly without
# being converted into numpy array.
def tanh(x):
    return np.tanh(x)

# Skip x in backward function and return the gradient of x
# Tensor must be actively converted to numpy array, otherwise,
# operations such as +/- can't be used.
def tanh_grad(y, dy):
    return np.array(dy) * (1 - np.square(np.array(y)))

# Creates a forward function for debugging running networks(print value)
def debug_func(x):
    print(x)

def create_tmp_var(name, dtype, shape):
    return paddle.static.default_main_program().current_block().create_var(
        name=name, dtype=dtype, shape=shape)

def simple_net(img, label):
    hidden = img
    for idx in six.moves.range(4):
        hidden = paddle.static.nn.fc(hidden, size=200)
        new_hidden = create_tmp_var(name='hidden_{}'.format(idx),
            dtype=hidden.dtype, shape=hidden.shape)

        # User-defined forward and backward
        hidden = paddle.static.py_func(func=tanh, x=hidden,
            out=new_hidden, backward_func=tanh_grad,
            skip_vars_in_backward_input=hidden)

        # User-defined debug functions that print out the input Tensor
        paddle.static.py_func(func=debug_func, x=hidden, out=None)

    prediction = paddle.static.nn.fc(hidden, size=10, activation='softmax')
    ce_loss = paddle.nn.loss.CrossEntropyLoss()
    return ce_loss(prediction, label)

x = paddle.static.data(name='x', shape=[1,4], dtype='float32')
y = paddle.static.data(name='y', shape=[1,10], dtype='int64')
res = simple_net(x, y)

exe = paddle.static.Executor(paddle.CPUPlace())
exe.run(paddle.static.default_startup_program())
input1 = np.random.random(size=[1,4]).astype('float32')
input2 = np.random.randint(1, 10, size=[1,10], dtype='int64')
out = exe.run(paddle.static.default_main_program(),
              feed={'x':input1, 'y':input2},
              fetch_list=[res.name])
print(out)