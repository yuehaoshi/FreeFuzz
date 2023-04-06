import paddle

linears = paddle.nn.LayerList([paddle.nn.Linear(10, 10) for i in range(10)])
another = paddle.nn.Linear(10, 10)
linears.insert(3, another)
print(linears[3] is another)  # True
another = paddle.nn.Linear(10, 10)
linears.insert(-1, another)
print(linears[-2] is another) # True