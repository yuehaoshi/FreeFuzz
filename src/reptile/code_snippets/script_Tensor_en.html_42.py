import paddle

x = paddle.to_tensor(1.0, stop_gradient=False)
detach_x = x.detach()
detach_x[:] = 10.0
print(x)  # Tensor(shape=[1], dtype=float32, place=CPUPlace, stop_gradient=False,
          #        [10.])
y = x**2
y.backward()
print(x.grad)         # [20.0]
print(detach_x.grad)  # None, 'stop_gradient=True' by default

detach_x.stop_gradient = False # Set stop_gradient to be False, supported auto-grad
z = detach_x**3
z.backward()

print(x.grad)         # [20.0], detach_x is detached from x's graph, not affect each other
print(detach_x.grad)  # [300.0], detach_x has its own graph

# Due to sharing of data with origin Tensor, There are some unsafe operations:
y = 2 * x
detach_x[:] = 5.0
y.backward()
# It will raise Error:
#   one of the variables needed for gradient computation has been modified by an inplace operation.