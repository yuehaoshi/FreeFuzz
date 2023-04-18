import paddle

emb = paddle.nn.Embedding(10, 10)

state_dict = emb.state_dict()
paddle.save(state_dict, "paddle_dy.pdparams")
para_state_dict = paddle.load("paddle_dy.pdparams")
emb.set_state_dict(para_state_dict)