import paddle
import paddle.static as static

# required: gpu

paddle.enable_static()

cuda_places = static.cuda_places()