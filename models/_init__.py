from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AutoTokenizer, AutoConfig, AutoModel

from torch.nn import Linear
from models.network.TextCNN import TextCNN

encoders_dict = {
    'bert': (BertTokenizer, BertConfig, BertModel),
    'auto': (AutoTokenizer, AutoConfig, AutoModel)
}

targets_dict = {
    'linear': Linear,
    'textcnn': TextCNN
}
