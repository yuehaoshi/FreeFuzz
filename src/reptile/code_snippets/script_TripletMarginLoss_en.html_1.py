import paddle

input = paddle.to_tensor([[1, 5, 3], [0, 3, 2], [1, 4, 1]], dtype=paddle.float32)
positive= paddle.to_tensor([[5, 1, 2], [3, 2, 1], [3, -1, 1]], dtype=paddle.float32)
negative = paddle.to_tensor([[2, 1, -3], [1, 1, -1], [4, -2, 1]], dtype=paddle.float32)
triplet_margin_loss = paddle.nn.TripletMarginLoss(reduction='none')
loss = triplet_margin_loss(input, positive, negative)
print(loss)
# Tensor([0.        , 0.57496738, 0.        ])

triplet_margin_loss = paddle.nn.TripletMarginLoss(margin=1.0, swap=True, reduction='mean', )
loss = triplet_margin_loss(input, positive, negative,)
print(loss)
# Tensor([0.19165580])