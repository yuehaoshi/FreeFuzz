import numpy as np
import paddle

x = (np.array([[3,2,3],[2, 2, 3]]) + 1j * np.array([[3,2,3],[2, 2, 3]])).astype(np.complex128)
xp = paddle.to_tensor(x)
hfft2_xp = paddle.fft.hfft2(xp).numpy()
print(hfft2_xp)
#  [[19.  7.  3. -9.]
#   [ 1.  1.  1.  1.]]