import paddle

input = paddle.to_tensor([[0.5, 0.6, 0.7],[0.3, 0.5, 0.2]], 'float32')
label = paddle.to_tensor([[1.0, -1.0, 1.0],[-1.0, 1.0, 1.0]], 'float32')
soft_margin_loss = paddle.nn.SoftMarginLoss()
output = soft_margin_loss(input, label)
print(output)
# Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
#        [0.64022040])

input_np = paddle.uniform(shape=(5, 5), min=0.1, max=0.8, dtype="float64")
label_np = paddle.randint(high=2, shape=(5, 5), dtype="int64")
label_np[label_np==0]=-1
input = paddle.to_tensor(input_np)
label = paddle.to_tensor(label_np)
soft_margin_loss = paddle.nn.SoftMarginLoss(reduction='none')
output = soft_margin_loss(input, label)
print(output)
# Tensor(shape=[5, 5], dtype=float64, place=Place(gpu:0), stop_gradient=True,
#        [[0.61739663, 0.51405668, 1.09346100, 0.42385561, 0.91602303],
#         [0.76997038, 1.01977148, 0.98971722, 1.13976032, 0.88152088],
#         [0.55476735, 1.10505384, 0.89923519, 0.45018155, 1.06587511],
#         [0.37998142, 0.48067240, 0.47791212, 0.55664053, 0.98581399],
#         [0.78571653, 0.59319711, 0.39701841, 0.76172109, 0.83781742]])