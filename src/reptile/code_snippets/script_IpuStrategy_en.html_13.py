# required: ipu

import paddle
import paddle.static as static

paddle.enable_static()

ipu_strategy = static.IpuStrategy()
ipu_strategy.disable_pattern("ViewSimplifyPattern")