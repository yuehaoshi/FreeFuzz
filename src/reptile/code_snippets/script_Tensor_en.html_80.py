import paddle

x = paddle.to_tensor(1.)
print(x.is_leaf) # True

x = paddle.to_tensor(1., stop_gradient=True)
y = x + 1
print(x.is_leaf) # True
print(y.is_leaf) # True

x = paddle.to_tensor(1., stop_gradient=False)
y = x + 1
print(x.is_leaf) # True
print(y.is_leaf) # False