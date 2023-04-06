from paddle.text.datasets import WMT16
wmt16 = WMT16(mode='train', src_dict_size=50, trg_dict_size=50)
en_dict = wmt16.get_dict('en')