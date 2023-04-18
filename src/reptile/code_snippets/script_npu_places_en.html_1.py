# required: npu

import paddle
import paddle.static as static

paddle.enable_static()
npu_places = static.npu_places()