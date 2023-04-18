import paddle

linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
another = paddle.nn.Linear(10, 10)
linears.append(another)
print(len(linears))  # 11