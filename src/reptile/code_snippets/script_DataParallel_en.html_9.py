import paddle

linear1 = paddle.nn.Linear(10, 3)
linear2 = paddle.nn.Linear(3, 10, bias_attr=False)
model = paddle.nn.Sequential(linear1, linear2)

layer_list = list(model.children())

print(layer_list)   # [<paddle.nn.layer.common.Linear object at 0x7f7b8113f830>, <paddle.nn.layer.common.Linear object at 0x7f7b8113f950>]