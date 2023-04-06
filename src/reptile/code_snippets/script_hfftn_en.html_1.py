import numpy as np
import paddle

x = (np.array([2, 2, 3]) + 1j * np.array([2, 2, 3])).astype(np.complex128)
xp = paddle.to_tensor(x)
hfftn_xp = paddle.fft.hfftn(xp).numpy()
print(hfftn_xp)
#  [ 9.  3.  1. -5.]