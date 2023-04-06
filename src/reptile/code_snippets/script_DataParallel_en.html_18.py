import paddle

fc1 = paddle.nn.Linear(10, 3)
fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
model = paddle.nn.Sequential(fc1, fc2)
for name, param in model.named_parameters():
    print(name, param)