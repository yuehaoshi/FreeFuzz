import paddle
arg_1_tensor = paddle.rand([3, 2], dtype=paddle.float64)
arg_1 = arg_1_tensor.clone()
int_tensor = paddle.randint(low=0, high=255, shape=[2], dtype='int32')
uint8_tensor = int_tensor.astype('uint8')
arg_2_tensor = uint8_tensor
arg_2 = arg_2_tensor.clone()
res = paddle.linalg.lu_unpack(arg_1,arg_2,)
