import paddle
int_tensor = paddle.randint(low=-128, high=127, shape=[6], dtype='int32')
int8_tensor = int_tensor.astype('int8')
arg_1_tensor = int8_tensor
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-64,32768,[6], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
arg_3_tensor = paddle.randint(-16,8,[6], dtype=paddle.int64)
arg_3 = arg_3_tensor.clone()
res = paddle.where(arg_1,arg_2,arg_3,)
