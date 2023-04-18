import paddle

linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
another_list = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(5)])
linears.extend(another_list)
print(len(linears))  # 15
print(another_list[0] is linears[10])  # True