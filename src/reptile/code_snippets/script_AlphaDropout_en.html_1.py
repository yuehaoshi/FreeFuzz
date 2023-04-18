import paddle

x = paddle.to_tensor([[-1, 1], [-1, 1]], dtype="float32")
m = paddle.nn.AlphaDropout(p=0.5)
y_train = m(x)
print(y_train)
# Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[-0.77919382,  1.66559887],
#         [-0.77919382, -0.77919382]])

m.eval()  # switch the model to test phase
y_test = m(x)
print(y_test)
# Tensor(shape=[2, 2], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [[-1.,  1.],
#         [-1.,  1.]])