.. code-block:: python
    import paddle
    import paddle.nn as nn

    model = nn.Sequential(nn.Conv2D(3, 5, 3), nn.BatchNorm2D(5))
    sync_model = nn.SyncBatchNorm.convert_sync_batchnorm(model)