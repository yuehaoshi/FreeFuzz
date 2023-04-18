# required: gpu
import paddle

paddle.enable_static()
support_gpu = paddle.is_compiled_with_cuda()
place = paddle.CPUPlace()
if support_gpu:
    place = paddle.CUDAPlace(0)

# if GPU is supported, the three OPs below will be automatically assigned to CUDAPlace(0)
data1 = paddle.full(shape=[1, 3, 8, 8], fill_value=0.5, dtype='float32')
data2 = paddle.full(shape=[1, 3, 64], fill_value=0.5, dtype='float32')
shape = paddle.shape(data2)

with paddle.static.device_guard("cpu"):
    # Ops created here will be placed on CPUPlace
    shape = paddle.slice(shape, axes=[0], starts=[0], ends=[4])
with paddle.static.device_guard('gpu'):
    # if GPU is supported, OPs created here will be placed on CUDAPlace(0), otherwise on CPUPlace
    out = paddle.reshape(data1, shape=shape)

exe = paddle.static.Executor(place)
exe.run(paddle.static.default_startup_program())
result = exe.run(fetch_list=[out])