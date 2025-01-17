import paddle
import multiprocessing
from contextlib import closing
import socket

port_set = set()

def find_free_port():
    def _free_port():
        with closing(socket.socket(socket.AF_INET,
            socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]
    while True:
        port = _free_port()
        if port not in port_set:
            port_set.add(port)
            return port

def test_gloo_barrier(id, rank_num, server_endpoint):
    paddle.distributed.gloo_init_parallel_env(
        id, rank_num, server_endpoint)
    paddle.distributed.gloo_barrier()

def test_gloo_barrier_with_multiprocess(num_of_ranks):
    jobs = []
    server_endpoint = "127.0.0.1:%s" % (find_free_port())
    for id in range(num_of_ranks):
        p = multiprocessing.Process(
            target=test_gloo_barrier,
            args=(id, num_of_ranks, server_endpoint))
        jobs.append(p)
        p.start()
    for proc in jobs:
        proc.join()

if __name__ == '__main__':
    # Arg: number of ranks (processes)
    test_gloo_barrier_with_multiprocess(2)