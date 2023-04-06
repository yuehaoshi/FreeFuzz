# required: gpu
import paddle
x = paddle.to_tensor(1.0, place=paddle.CPUPlace())
print(x.place)        # CPUPlace

y = x.cuda()
print(y.place)        # CUDAPlace(0)

y = x.cuda(None)
print(y.place)        # CUDAPlace(0)

y = x.cuda(1)
print(y.place)        # CUDAPlace(1)