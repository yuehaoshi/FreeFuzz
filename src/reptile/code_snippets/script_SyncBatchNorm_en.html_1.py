import paddle
import paddle.nn as nn
import numpy as np

x = np.array([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')
x = paddle.to_tensor(x)

if paddle.is_compiled_with_cuda():
    sync_batch_norm = nn.SyncBatchNorm(2)
    hidden1 = sync_batch_norm(x)
    print(hidden1)
    # [[[[0.26824948, 1.0936325],[0.26824948, -1.6301316]],[[ 0.8095662, -0.665287],[-1.2744656, 1.1301866 ]]]]