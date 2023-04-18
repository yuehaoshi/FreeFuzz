import paddle

emb = paddle.nn.Embedding(10, 10)

state_dict = emb.to_static_state_dict()
paddle.save( state_dict, "paddle_dy.pdparams")