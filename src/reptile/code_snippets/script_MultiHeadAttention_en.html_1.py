import paddle

# encoder input: [batch_size, sequence_length, d_model]
query = paddle.rand((2, 4, 128))
# self attention mask: [batch_size, num_heads, query_len, query_len]
attn_mask = paddle.rand((2, 2, 4, 4))
multi_head_attn = paddle.nn.MultiHeadAttention(128, 2)
output = multi_head_attn(query, None, None, attn_mask=attn_mask)  # [2, 4, 128]