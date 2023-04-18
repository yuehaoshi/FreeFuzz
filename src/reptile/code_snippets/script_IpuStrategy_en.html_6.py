# required: ipu

import paddle
import paddle.static as static

paddle.enable_static()

ipu_strategy = static.IpuStrategy()
ipu_strategy.set_graph_config(num_ipus=1,
                            is_training=True,
                            micro_batch_size=1,
                            enable_manual_shard=False)