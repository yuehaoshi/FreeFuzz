import paddle

paddle.enable_static()
place = paddle.CPUPlace() # you can set place = paddle.CUDAPlace(0) to use gpu
exe = paddle.static.Executor(place)
x = paddle.static.data(name="x", shape=[None, 10, 10], dtype="int64")
y = paddle.static.data(name="y", shape=[None, 1], dtype="int64", lod_level=1)
dataset = paddle.fluid.DatasetFactory().create_dataset()
dataset.set_use_var([x, y])
dataset.set_thread(1)
# you should set your own filelist, e.g. filelist = ["dataA.txt"]
filelist = []
dataset.set_filelist(filelist)
exe.run(paddle.static.default_startup_program())
exe.train_from_dataset(program=paddle.static.default_main_program(),
                       dataset=dataset)