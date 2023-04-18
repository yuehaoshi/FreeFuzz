import paddle

linear = paddle.nn.Linear(1,1)
print(linear.parameters())  # print linear_0.w_0 and linear_0.b_0