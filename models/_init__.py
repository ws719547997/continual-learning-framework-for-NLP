from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AutoTokenizer, AutoConfig, AutoModel

from torch.nn import Linear
from network.TextCNN import TextCNN

models = {
    'bert': (BertTokenizer, BertConfig, BertModel),
    'auto': (AutoTokenizer, AutoConfig, AutoModel)
}

top = {
    'linear': Linear,
    'textcnn': TextCNN
}
