import paddle
arg_class = paddle.distributed.InMemoryDataset()
arg_1 = "replicate"
res = arg_class(*arg_1)
