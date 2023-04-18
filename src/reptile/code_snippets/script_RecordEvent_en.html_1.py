import paddle
import paddle.profiler as profiler
# method1: using context manager
with profiler.RecordEvent("record_add"):
    data1 = paddle.randn(shape=[3])
    data2 = paddle.randn(shape=[3])
    result = data1 + data2
# method2: call begin() and end()
record_event = profiler.RecordEvent("record_add")
record_event.begin()
data1 = paddle.randn(shape=[3])
data2 = paddle.randn(shape=[3])
result = data1 + data2
record_event.end()