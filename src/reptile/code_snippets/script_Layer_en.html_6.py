import paddle
import numpy as np

# the forward_post_hook change the output of the layer: output = output * 2
def forward_post_hook(layer, input, output):
    # user can use layer, input and output for information statistis tasks

    # change the output
    return output * 2

linear = paddle.nn.Linear(13, 5)

# register the hook
forward_post_hook_handle = linear.register_forward_post_hook(forward_post_hook)

value1 = np.arange(26).reshape(2, 13).astype("float32")
in1 = paddle.to_tensor(value1)

out0 = linear(in1)

# remove the hook
forward_post_hook_handle.remove()

out1 = linear(in1)

# hook change the linear's output to output * 2, so out0 is equal to out1 * 2.
assert (out0.numpy() == (out1.numpy()) * 2).any()