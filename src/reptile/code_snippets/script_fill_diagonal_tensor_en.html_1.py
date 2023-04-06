import paddle

x = paddle.ones((4, 3)) * 2
y = paddle.ones((3,))
nx = x.fill_diagonal_tensor(y)
print(nx.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]