import paddle

linear1 = paddle.nn.Linear(10, 3)
linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
model = paddle.nn.Sequential(linear1, linear2)
for prefix, layer in model.named_children():
    print(prefix, layer)
    # ('0', <paddle.nn.layer.common.Linear object at 0x7fb61ed85830>)
    # ('1', <paddle.nn.layer.common.Linear object at 0x7fb61ed85950>)