# Example1: set Regularizer in optimizer
import paddle
from paddle.regularizer import L2Decay
linear = paddle.nn.Linear(10, 10)
inp = paddle.rand(shape=[10, 10], dtype="float32")
out = linear(inp)
loss = paddle.mean(out)
beta1 = paddle.to_tensor([0.9], dtype="float32")
beta2 = paddle.to_tensor([0.99], dtype="float32")
momentum = paddle.optimizer.Momentum(
    learning_rate=0.1,
    parameters=linear.parameters(),
    weight_decay=L2Decay(0.0001))
back = out.backward()
momentum.step()
momentum.clear_grad()

# Example2: set Regularizer in parameters
# Set L2 regularization in parameters.
# Global regularizer does not take effect on my_conv2d for this case.
from paddle.nn import Conv2D
from paddle import ParamAttr
from paddle.regularizer import L2Decay

my_conv2d = Conv2D(
        in_channels=10,
        out_channels=10,
        kernel_size=1,
        stride=1,
        padding=0,
        weight_attr=ParamAttr(regularizer=L2Decay(coeff=0.01)),
        bias_attr=False)