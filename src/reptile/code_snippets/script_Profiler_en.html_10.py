# required: gpu
import paddle.profiler as profiler
prof = profiler.Profiler(
    targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU],
    scheduler = (3, 7),
    on_trace_ready = profiler.export_chrome_tracing('./log'))
prof.start()
for iter in range(10):
    #train()
    prof.step()
prof.stop()
prof.summary(sorted_by=profiler.SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')