import paddle

x = paddle.to_tensor([1 + 2j, 3 + 4j])
print(paddle.is_complex(x))
# True

x = paddle.to_tensor([1.1, 1.2])
print(paddle.is_complex(x))
# False

x = paddle.to_tensor([1, 2, 3])
print(paddle.is_complex(x))
# False