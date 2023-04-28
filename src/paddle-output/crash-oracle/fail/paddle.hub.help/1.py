import paddle
arg_1 = "lyuwenyu/paddlehub_demo:main"
arg_2 = "zeros"
arg_3 = -54
res = paddle.hub.help(arg_1,model=arg_2,source=arg_3,)
