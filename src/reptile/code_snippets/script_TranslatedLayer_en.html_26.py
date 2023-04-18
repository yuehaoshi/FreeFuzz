import paddle

emb = paddle.nn.Embedding(10, 10)

state_dict = emb.state_dict()
paddle.save( state_dict, "paddle_dy.pdparams")