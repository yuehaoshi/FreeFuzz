import paddle
arg_1_tensor = paddle.randint(-1024,4,[3], dtype=paddle.int64)
arg_1 = arg_1_tensor.clone()
arg_2_tensor = paddle.randint(-32768,64,[7], dtype=paddle.int64)
arg_2 = arg_2_tensor.clone()
float_tensor = paddle.rand([18], 'float32')
f16_tensor = float_tensor.astype('float16')
arg_3_tensor = f16_tensor
arg_3 = arg_3_tensor.clone()
res = paddle.geometric.reindex_graph(arg_1,arg_2,arg_3,)
