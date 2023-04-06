import paddle
from paddle.autograd import PyLayer

class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        # ctx is a object of PyLayerContext.
        y = paddle.tanh(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        # ctx is a object of PyLayerContext.
        y, = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(y))
        return grad