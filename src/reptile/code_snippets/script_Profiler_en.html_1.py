# required: gpu
import paddle.profiler as profiler
with profiler.Profiler(
        targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
        scheduler = (2, 5),
        on_trace_ready = profiler.export_chrome_tracing('./log')) as p:
    for iter in range(10):
        #train()
        p.step()