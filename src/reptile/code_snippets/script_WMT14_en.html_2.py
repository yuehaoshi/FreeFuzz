from paddle.text.datasets import WMT14
wmt14 = WMT14(mode='train', dict_size=50)
src_dict, trg_dict = wmt14.get_dict()