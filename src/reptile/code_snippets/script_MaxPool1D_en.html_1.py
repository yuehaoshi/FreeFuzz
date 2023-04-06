import paddle
import paddle.nn as nn
import numpy as np

data = paddle.to_tensor(np.random.uniform(-1, 1, [1, 3, 32]).astype(np.float32))
MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0)
pool_out = MaxPool1D(data)
# pool_out shape: [1, 3, 16]

MaxPool1D = nn.MaxPool1D(kernel_size=2, stride=2, padding=0, return_mask=True)
pool_out, indices = MaxPool1D(data)
# pool_out shape: [1, 3, 16], indices shape: [1, 3, 16]