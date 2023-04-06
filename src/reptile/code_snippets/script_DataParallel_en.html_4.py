import paddle
import paddle.distributed as dist

dist.init_parallel_env()

emb = fluid.dygraph.Embedding([10, 10])
emb = fluid.dygraph.DataParallel(emb)

state_dict = emb.state_dict()
paddle.save(state_dict, "paddle_dy.pdparams")