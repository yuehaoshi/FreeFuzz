import paddle
paddle.enable_static()

sparse_feature_dim = 1024
embedding_size = 64

shows = paddle.static.data(name='show', shape=[1], dtype='int64')
clicks = paddle.static.data(name='click', shape=[1], dtype='int64')
input = paddle.static.data(name='ins', shape=[1], dtype='int64')

entry = paddle.distributed.ShowClickEntry("show", "click")

emb = paddle.static.nn.sparse_embedding(
    input=input,
    size=[sparse_feature_dim, embedding_size],
    is_test=False,
    entry=entry,
    param_attr=paddle.ParamAttr(name="SparseFeatFactors",
                               initializer=paddle.nn.initializer.Uniform()))