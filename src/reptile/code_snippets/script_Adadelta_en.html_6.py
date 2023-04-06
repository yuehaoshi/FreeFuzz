import paddle

emb = paddle.nn.Embedding(10, 10)

layer_state_dict = emb.state_dict()
paddle.save(layer_state_dict, "emb.pdparams")

scheduler = paddle.optimizer.lr.NoamDecay(
    d_model=0.01, warmup_steps=100, verbose=True)
adam = paddle.optimizer.Adam(
    learning_rate=scheduler,
    parameters=emb.parameters())
opt_state_dict = adam.state_dict()
paddle.save(opt_state_dict, "adam.pdopt")

opti_state_dict = paddle.load("adam.pdopt")
adam.set_state_dict(opti_state_dict)