import paddle
arg_1 = -40
arg_2 = "mean"
res = paddle.nn.CTCLoss(blank=arg_1,reduction=arg_2,)