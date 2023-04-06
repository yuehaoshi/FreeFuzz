import numpy as np
import paddle

x = (np.array([[3,2,3],[2, 2, 3]]) + 1j * np.array([[3,2,3],[2, 2, 3]])).astype(np.complex128)
xp = paddle.to_tensor(x)
irfft2_xp = paddle.fft.irfft2(xp).numpy()
print(irfft2_xp)
#  [[ 2.375 -1.125  0.375  0.875]
#   [ 0.125  0.125  0.125  0.125]]