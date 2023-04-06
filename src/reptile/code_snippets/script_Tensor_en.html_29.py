import paddle
input = paddle.uniform([10, 2])
linear = paddle.nn.Linear(2, 3)
out = linear(input)
out.backward()
print("Before clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))
linear.weight.clear_gradient()
print("After clear_gradient, linear.weight.grad: {}".format(linear.weight.grad))