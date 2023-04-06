import numpy as np
import paddle

x = np.array([1, -1j, -1])
xp = paddle.to_tensor(x)
hfft_xp = paddle.fft.hfft(xp).numpy()
print(hfft_xp)
#  [0. 0. 0. 4.]