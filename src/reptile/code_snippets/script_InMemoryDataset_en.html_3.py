import paddle
import os
paddle.enable_static()

with open("test_queue_dataset_run_a.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3"
    f.write(data)
with open("test_queue_dataset_run_b.txt", "w") as f:
    data = "2 1 2 2 5 4 2 2 7 2 1 3"
    f.write(data)

slots = ["slot1", "slot2", "slot3", "slot4"]
slots_vars = []
for slot in slots:
    var = paddle.static.data(
        name=slot, shape=[None, 1], dtype="int64", lod_level=1)
    slots_vars.append(var)

dataset = paddle.distributed.InMemoryDataset()
dataset.init(
    batch_size=1,
    thread_num=2,
    input_type=1,
    pipe_command="cat",
    use_var=slots_vars)
dataset.set_filelist(
    ["test_queue_dataset_run_a.txt", "test_queue_dataset_run_b.txt"])
dataset.load_into_memory()

place = paddle.CPUPlace()
exe = paddle.static.Executor(place)
startup_program = paddle.static.Program()
main_program = paddle.static.Program()
exe.run(startup_program)

exe.train_from_dataset(main_program, dataset)

os.remove("./test_queue_dataset_run_a.txt")
os.remove("./test_queue_dataset_run_b.txt")