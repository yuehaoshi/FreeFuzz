# required: gpu

import paddle
import paddle.nn as nn

x = paddle.to_tensor([[[[0.3, 0.4], [0.3, 0.07]], [[0.83, 0.37], [0.18, 0.93]]]]).astype('float32')

if paddle.is_compiled_with_cuda():
    sync_batch_norm = nn.SyncBatchNorm(2)
    hidden1 = sync_batch_norm(x)
    print(hidden1)
    # Tensor(shape=[1, 2, 2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=False,
    #        [[[[ 0.26824948,  1.09363246],
    #           [ 0.26824948, -1.63013160]],

    #          [[ 0.80956620, -0.66528702],
    #           [-1.27446556,  1.13018656]]]])