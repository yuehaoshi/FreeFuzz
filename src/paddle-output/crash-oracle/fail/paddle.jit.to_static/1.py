import paddle
arg_1 = "with_common_value"
res = paddle.jit.to_static(arg_1,)
