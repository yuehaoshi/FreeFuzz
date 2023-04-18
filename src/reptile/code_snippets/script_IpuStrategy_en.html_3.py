# required: ipu

import paddle
import paddle.static as static

ipu_strategy = static.IpuStrategy()

ipu_strategy.release_patch()