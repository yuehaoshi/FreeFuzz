from paddle.static import InputSpec

x_spec = InputSpec(shape=[4, 64], dtype='float32', name='x')
x_spec.unbatch()
print(x_spec) # InputSpec(shape=(64,), dtype=VarType.FP32, name=x)