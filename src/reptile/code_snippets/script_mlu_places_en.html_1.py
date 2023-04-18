# required: mlu

import paddle
import paddle.static as static

paddle.enable_static()
mlu_places = static.mlu_places()