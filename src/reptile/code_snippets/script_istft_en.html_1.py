import numpy as np
import paddle
from paddle.signal import stft, istft

paddle.seed(0)

# STFT
x = paddle.randn([8, 48000], dtype=paddle.float64)
y = stft(x, n_fft=512)  # [8, 257, 376]

# ISTFT
x_ = istft(y, n_fft=512)  # [8, 48000]

np.allclose(x, x_)  # True