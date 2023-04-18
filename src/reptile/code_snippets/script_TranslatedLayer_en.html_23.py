import paddle
import numpy as np

# the forward_pre_hook change the input of the layer: input = input * 2
def forward_pre_hook(layer, input):
    # user can use layer and input for information statistis tasks

    # change the input
    input_return = (input[0] * 2)
    return input_return

linear = paddle.nn.Linear(13, 5)

# register the hook
forward_pre_hook_handle = linear.register_forward_pre_hook(forward_pre_hook)

value0 = np.arange(26).reshape(2, 13).astype("float32")
in0 = paddle.to_tensor(value0)
out0 = linear(in0)

# remove the hook
forward_pre_hook_handle.remove()

value1 = value0 * 2
in1 = paddle.to_tensor(value1)
out1 = linear(in1)

# hook change the linear's input to input * 2, so out0 is equal to out1.
assert (out0.numpy() == out1.numpy()).any()