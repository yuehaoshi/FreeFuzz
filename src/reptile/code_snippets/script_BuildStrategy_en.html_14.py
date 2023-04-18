import paddle
import paddle.static as static

paddle.enable_static()

build_strategy = static.BuildStrategy()
build_strategy.remove_unnecessary_lock = True