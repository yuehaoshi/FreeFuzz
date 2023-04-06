from paddle.text.datasets import Conll05st

conll05st = Conll05st()
word_dict, predicate_dict, label_dict = conll05st.get_dict()