import paddle

# In imperative mode:
# size x: (2, 2, 3) and y: (2, 3, 2)
x = paddle.to_tensor([[[1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0]],
                    [[3.0, 3.0, 3.0],
                    [4.0, 4.0, 4.0]]])
y = paddle.to_tensor([[[1.0, 1.0],[2.0, 2.0],[3.0, 3.0]],
                    [[4.0, 4.0],[5.0, 5.0],[6.0, 6.0]]])
out = paddle.bmm(x, y)
#output size: (2, 2, 2)
#output value:
#[[[6.0, 6.0],[12.0, 12.0]],[[45.0, 45.0],[60.0, 60.0]]]
out_np = out.numpy()