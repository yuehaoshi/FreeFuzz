import paddle
from paddle.static import InputSpec

paddle.disable_static()

x = paddle.ones([2, 2], dtype="float32")
x_spec = InputSpec.from_tensor(x, name='x')
print(x_spec)  # InputSpec(shape=(2, 2), dtype=paddle.float32, name=x)