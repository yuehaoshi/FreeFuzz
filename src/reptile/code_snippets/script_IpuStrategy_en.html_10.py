# required: ipu

import paddle
import paddle.static as static

paddle.enable_static()

ipu_strategy = static.IpuStrategy()
options = {'num_ipus':1, 'enable_fp16': True}
ipu_strategy.set_options(options)