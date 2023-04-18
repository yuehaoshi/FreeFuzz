from paddle.static import InputSpec

input = InputSpec([None, 784], 'float32', 'x')
label = InputSpec([None, 1], 'int64', 'label')

print(input)  # InputSpec(shape=(-1, 784), dtype=paddle.float32, name=x)
print(label)  # InputSpec(shape=(-1, 1), dtype=paddle.int64, name=label)