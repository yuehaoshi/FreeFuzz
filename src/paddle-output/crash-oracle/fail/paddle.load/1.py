import paddle
arg_1 = "/Users/huyiteng/.paddlenlp/models/__internal_testing__/tiny-random-ernie/model_state.pdparams"
arg_2 = True
res = paddle.load(arg_1,return_numpy=arg_2,)
