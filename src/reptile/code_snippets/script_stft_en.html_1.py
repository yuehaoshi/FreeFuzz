import paddle
from paddle.signal import stft

# real-valued input
x = paddle.randn([8, 48000], dtype=paddle.float64)
y1 = stft(x, n_fft=512)  # [8, 257, 376]
y2 = stft(x, n_fft=512, onesided=False)  # [8, 512, 376]

# complex input
x = paddle.randn([8, 48000], dtype=paddle.float64) + \
        paddle.randn([8, 48000], dtype=paddle.float64)*1j  # [8, 48000] complex128
y1 = stft(x, n_fft=512, center=False, onesided=False)  # [8, 512, 372]