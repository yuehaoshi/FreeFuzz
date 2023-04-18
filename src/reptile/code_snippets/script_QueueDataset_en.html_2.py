import paddle
dataset = paddle.distributed.fleet.DatasetBase()
dataset.set_filelist(['a.txt', 'b.txt'])