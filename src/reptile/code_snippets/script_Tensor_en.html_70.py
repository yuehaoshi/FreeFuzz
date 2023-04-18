import paddle

tensor = paddle.to_tensor([0, 1, 2, 3, 4])

tensor.fill_(0)
print(tensor.tolist())   #[0, 0, 0, 0, 0]