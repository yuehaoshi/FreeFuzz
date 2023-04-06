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
print dataset.get_memory_data_size()