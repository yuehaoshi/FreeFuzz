import paddle
import paddle.profiler as profiler
record_event = profiler.RecordEvent("record_mul")
record_event.begin()
data1 = paddle.randn(shape=[3])
data2 = paddle.randn(shape=[3])
result = data1 * data2
record_event.end()