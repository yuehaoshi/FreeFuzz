import numpy as np
from paddle.static import InputSpec

x = np.ones([2, 2], np.float32)
x_spec = InputSpec.from_numpy(x, name='x')
print(x_spec)  # InputSpec(shape=(2, 2), dtype=paddle.float32, name=x)