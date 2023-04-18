import paddle
import paddle.static as static

paddle.enable_static()

build_strategy = static.BuildStrategy()
build_strategy.fuse_broadcast_ops = True