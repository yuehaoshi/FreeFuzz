import paddle
arg_1_tensor = paddle.rand([3, 3], dtype=paddle.float32)
arg_1 = arg_1_tensor.clone()
arg_2 = "max"
arg_class = paddle.text.ViterbiDecoder(arg_1,include_bos_eos_tag=arg_2,)
arg_3 = None
res = arg_class(*arg_3)
