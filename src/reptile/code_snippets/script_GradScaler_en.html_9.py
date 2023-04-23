# required: gpu,xpu
import paddle
scaler = paddle.amp.GradScaler(enable=True,
                               init_loss_scaling=1024,
                               incr_ratio=2.0,
                               decr_ratio=0.5,
                               incr_every_n_steps=1000,
                               decr_every_n_nan_or_inf=2,
                               use_dynamic_loss_scaling=True)
init_loss_scaling = scaler.get_init_loss_scaling()
print(init_loss_scaling) # 1024