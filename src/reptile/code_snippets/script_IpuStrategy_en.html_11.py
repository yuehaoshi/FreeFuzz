# required: ipu

import paddle
import paddle.static as static

paddle.enable_static()

ipu_strategy = static.IpuStrategy()
num_ipus = ipu_strategy.get_option('num_ipus')