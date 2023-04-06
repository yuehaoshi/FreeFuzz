import paddle
paddle.enable_static()

dataset = paddle.distributed.InMemoryDataset()
slots = ["slot1", "slot2", "slot3", "slot4"]
slots_vars = []
for slot in slots:
    var = paddle.static.data(
        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
    slots_vars.append(var)
dataset.init(
    batch_size=1,
    thread_num=2,
    input_type=1,
    pipe_command="cat",
    use_var=slots_vars)
filelist = ["a.txt", "b.txt"]
dataset.set_filelist(filelist)
dataset.load_into_memory()
dataset.global_shuffle()
exe = paddle.static.Executor(paddle.CPUPlace())
startup_program = paddle.static.Program()
main_program = paddle.static.Program()
exe.run(startup_program)
exe.train_from_dataset(main_program, dataset)
dataset.release_memory()