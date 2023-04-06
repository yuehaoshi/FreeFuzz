import paddle
from paddle.autograd import PyLayer

class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x, func1, func2=paddle.square):
        ctx.func = func2
        y = func1(x)
        # Pass tensors to backward.
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, dy):
        # Get the tensors passed by forward.
        y, = ctx.saved_tensor()
        grad = dy * (1 - ctx.func(y))
        return grad


data = paddle.randn([2, 3], dtype="float64")
data.stop_gradient = False
# run custom Layer.
z = cus_tanh.apply(data, func1=paddle.tanh)