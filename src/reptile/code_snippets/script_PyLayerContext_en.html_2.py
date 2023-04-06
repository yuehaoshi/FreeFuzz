import paddle
from paddle.autograd import PyLayer

class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        # ctx is a context object that store some objects for backward.
        y = paddle.tanh(x)
        # Pass tensors to backward.
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        # Get the tensors passed by forward.
        y, = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad