import paddle
x = paddle.to_tensor(1.0, place=paddle.CUDAPlace(0))
print(x.place)      # CUDAPlace(0)

y = x.pin_memory()
print(y.place)      # CUDAPinnedPlace