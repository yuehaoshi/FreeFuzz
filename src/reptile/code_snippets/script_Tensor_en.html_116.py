import paddle
import numpy as np
shape=[2, 3, 4]
np_input = np.arange(24).astype('float32') - 12
np_input = np_input.reshape(shape)
x = paddle.to_tensor(np_input)
#[[[-12. -11. -10.  -9.] [ -8.  -7.  -6.  -5.] [ -4.  -3.  -2.  -1.]]
# [[  0.   1.   2.   3.] [  4.   5.   6.   7.] [  8.   9.  10.  11.]]]

# compute frobenius norm along last two dimensions.
out_fro = paddle.norm(x, p='fro', axis=[0,1])
# out_fro.numpy() [17.435596 16.911535 16.7332   16.911535]

# compute 2-order vector norm along last dimension.
out_pnorm = paddle.norm(x, p=2, axis=-1)
#out_pnorm.numpy(): [[21.118711  13.190906   5.477226]
#                    [ 3.7416575 11.224972  19.131126]]

# compute 2-order  norm along [0,1] dimension.
out_pnorm = paddle.norm(x, p=2, axis=[0,1])
#out_pnorm.numpy(): [17.435596 16.911535 16.7332   16.911535]

# compute inf-order  norm
out_pnorm = paddle.norm(x, p=np.inf)
#out_pnorm.numpy()  = [12.]
out_pnorm = paddle.norm(x, p=np.inf, axis=0)
#out_pnorm.numpy(): [[12. 11. 10. 9.] [8. 7. 6. 7.] [8. 9. 10. 11.]]

# compute -inf-order  norm
out_pnorm = paddle.norm(x, p=-np.inf)
#out_pnorm.numpy(): [0.]
out_pnorm = paddle.norm(x, p=-np.inf, axis=0)
#out_pnorm.numpy(): [[0. 1. 2. 3.] [4. 5. 6. 5.] [4. 3. 2. 1.]]