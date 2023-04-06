import paddle

x = paddle.to_tensor([10000., 1e-07])
y = paddle.to_tensor([10000.1, 1e-08])
result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                        equal_nan=False, name="ignore_nan")
np_result1 = result1.numpy()
# [False]
result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                            equal_nan=True, name="equal_nan")
np_result2 = result2.numpy()
# [False]

x = paddle.to_tensor([1.0, float('nan')])
y = paddle.to_tensor([1.0, float('nan')])
result1 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                        equal_nan=False, name="ignore_nan")
np_result1 = result1.numpy()
# [False]
result2 = paddle.allclose(x, y, rtol=1e-05, atol=1e-08,
                            equal_nan=True, name="equal_nan")
np_result2 = result2.numpy()
# [True]