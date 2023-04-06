import numpy as np
import paddle

x = (np.array([2, 2, 3]) + 1j * np.array([2, 2, 3])).astype(np.complex128)
xp = paddle.to_tensor(x)
irfftn_xp = paddle.fft.irfftn(xp).numpy()
print(irfftn_xp)
#  [ 2.25 -1.25  0.25  0.75]