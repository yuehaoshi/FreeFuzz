import paddle
from paddle.nn import TransformerEncoderLayer, TransformerEncoder

# encoder input: [batch_size, src_len, d_model]
enc_input = paddle.rand((2, 4, 128))
# self attention mask: [batch_size, n_head, src_len, src_len]
attn_mask = paddle.rand((2, 2, 4, 4))
encoder_layer = TransformerEncoderLayer(128, 2, 512)
encoder = TransformerEncoder(encoder_layer, 2)
enc_output = encoder(enc_input, attn_mask)  # [2, 4, 128]