import paddle
x = paddle.rand((2,8,32,32))

spectral_norm = paddle.nn.SpectralNorm(x.shape, dim=1, power_iters=2)
spectral_norm_out = spectral_norm(x)

print(spectral_norm_out.shape) # [2, 8, 32, 32]