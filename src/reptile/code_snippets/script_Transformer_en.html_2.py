import paddle
from paddle.nn.layer.transformer import Transformer
length = 5
d_model, n_head, dim_feedforward = 8, 4, 64
transformer_paddle = Transformer(
    d_model, n_head, dim_feedforward=dim_feedforward)
mask = transformer_paddle.generate_square_subsequent_mask(length)
print(mask)

# [[  0. -inf -inf -inf -inf]
# [  0.   0. -inf -inf -inf]
# [  0.   0.   0. -inf -inf]
# [  0.   0.   0.   0. -inf]
# [  0.   0.   0.   0.   0.]]