import paddle

fc1 = paddle.nn.Linear(10, 3)
fc2 = paddle.nn.Linear(3, 10, bias_attr=False)
model = paddle.nn.Sequential(fc1, fc2)
for prefix, layer in model.named_sublayers():
    print(prefix, layer)