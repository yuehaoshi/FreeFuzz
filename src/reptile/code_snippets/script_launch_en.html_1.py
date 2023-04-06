.. code-block:: bash
    :name: code-block-example-bash5

   # To simulate distributed environment using single node, e.g., 2 servers and 4 workers, each worker use single gpu.

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m paddle.distributed.launch --server_num=2 --worker_num=4 train.py --lr=0.01