# required: ipu

import paddle
import paddle.static as static

paddle.enable_static()

ipu_strategy = static.IpuStrategy()
ipu_strategy.set_pipelining_config(enable_pipelining=False,
                                    batches_per_step=1,
                                    enable_gradient_accumulation=False,
                                    accumulation_factor=1)