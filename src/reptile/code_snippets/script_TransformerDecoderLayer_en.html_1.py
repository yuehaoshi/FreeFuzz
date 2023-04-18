import paddle
from paddle.nn import TransformerDecoderLayer

# decoder input: [batch_size, tgt_len, d_model]
dec_input = paddle.rand((2, 4, 128))
# encoder output: [batch_size, src_len, d_model]
enc_output = paddle.rand((2, 6, 128))
# self attention mask: [batch_size, n_head, tgt_len, tgt_len]
self_attn_mask = paddle.rand((2, 2, 4, 4))
# cross attention mask: [batch_size, n_head, tgt_len, src_len]
cross_attn_mask = paddle.rand((2, 2, 4, 6))
decoder_layer = TransformerDecoderLayer(128, 2, 512)
output = decoder_layer(dec_input,
                       enc_output,
                       self_attn_mask,
                       cross_attn_mask)  # [2, 4, 128]