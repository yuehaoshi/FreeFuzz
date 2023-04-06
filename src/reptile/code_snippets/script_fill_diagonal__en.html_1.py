.. code-block:: python
    import paddle
    x = paddle.ones((4, 3)) * 2
    x.fill_diagonal_(1.0)
    print(x.tolist())   #[[1.0, 2.0, 2.0], [2.0, 1.0, 2.0], [2.0, 2.0, 1.0], [2.0, 2.0, 2.0]]