import paddle
arg_1 = -17
arg_2 = "mean"
arg_class = paddle.nn.CTCLoss(blank=arg_1,reduction=arg_2,)
arg_3 = None
res = arg_class(*arg_3)
