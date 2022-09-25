from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AutoTokenizer, AutoConfig, AutoModel

from models.encoder.automodel import AutoPTM

from torch.nn import Linear
from models.network.TextCNN import TextCNN

encoders_args_dict = {
    'bert': (BertTokenizer, BertConfig, BertModel),
    'auto': (AutoTokenizer, AutoConfig, AutoModel)
}

encoders_dict = {
    'auto':AutoPTM
}

targets_dict = {
    'linear': Linear,
    'textcnn': TextCNN
}
