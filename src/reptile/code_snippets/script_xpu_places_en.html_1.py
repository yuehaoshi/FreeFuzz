# required: xpu

import paddle
import paddle.static as static

paddle.enable_static()
xpu_places = static.xpu_places()