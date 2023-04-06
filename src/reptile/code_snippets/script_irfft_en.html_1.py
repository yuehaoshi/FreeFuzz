import numpy as np
import paddle

x = np.array([1, -1j, -1])
xp = paddle.to_tensor(x)
irfft_xp = paddle.fft.irfft(xp).numpy()
print(irfft_xp)
#  [0. 1. 0. 0.]