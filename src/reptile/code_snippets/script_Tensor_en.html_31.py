import paddle

x = paddle.to_tensor(1.0, stop_gradient=False)
clone_x = x.clone()
y = clone_x**2
y.backward()
print(clone_x.stop_gradient) # False
print(clone_x.grad)          # [2.0], support gradient propagation
print(x.stop_gradient)       # False
print(x.grad)                # [2.0], clone_x support gradient propagation for x

x = paddle.to_tensor(1.0)
clone_x = x.clone()
clone_x.stop_gradient = False
z = clone_x**3
z.backward()
print(clone_x.stop_gradient) # False
print(clone_x.grad)          # [3.0], support gradient propagation
print(x.stop_gradient) # True
print(x.grad)          # None