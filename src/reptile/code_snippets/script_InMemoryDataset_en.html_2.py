import paddle
paddle.enable_static()

dataset = paddle.distributed.InMemoryDataset()
dataset.init(
    batch_size=1,
    thread_num=2,
    input_type=1,
    pipe_command="cat",
    use_var=[])
dataset._init_distributed_settings(
    parse_ins_id=True,
    parse_content=True,
    fea_eval=True,
    candidate_size=10000)
dataset.update_settings(batch_size=2)