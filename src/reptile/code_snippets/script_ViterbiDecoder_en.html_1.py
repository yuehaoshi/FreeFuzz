import paddle
paddle.seed(102)
batch_size, seq_len, num_tags = 2, 4, 3
emission = paddle.rand((batch_size, seq_len, num_tags), dtype='float32')
length = paddle.randint(1, seq_len + 1, [batch_size])
tags = paddle.randint(0, num_tags, [batch_size, seq_len])
transition = paddle.rand((num_tags, num_tags), dtype='float32')
decoder = paddle.text.ViterbiDecoder(transition, include_bos_eos_tag=False)
scores, path = decoder(emission, length) # scores: [3.37089300, 1.56825531], path: [[1, 0, 0], [1, 1, 0]]