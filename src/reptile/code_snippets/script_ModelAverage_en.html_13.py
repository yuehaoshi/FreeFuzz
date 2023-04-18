import paddle
emb = paddle.nn.Embedding(10, 10)

adam = paddle.optimizer.Adam(0.001, parameters=emb.parameters())
state_dict = adam.state_dict()