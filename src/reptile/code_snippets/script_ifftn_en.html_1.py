import numpy as np
import paddle

x = np.eye(3)
xp = paddle.to_tensor(x)
ifftn_xp = paddle.fft.ifftn(xp, axes=(1,)).numpy()
print(ifftn_xp)

#   [[ 0.33333333+0.j          0.33333333+0.j          0.33333333-0.j        ]
#   [ 0.33333333+0.j         -0.16666667+0.28867513j -0.16666667-0.28867513j]
#   [ 0.33333333+0.j         -0.16666667-0.28867513j -0.16666667+0.28867513j]]