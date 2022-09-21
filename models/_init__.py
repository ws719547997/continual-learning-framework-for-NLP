from transformers import BertTokenizer, BertConfig, BertModel
from transformers import AutoTokenizer, AutoConfig, AutoModel

from torch.nn import Linear

models = {
    'bert': (BertTokenizer, BertConfig, BertModel),
    'auto': (AutoTokenizer, AutoConfig, AutoModel)
}

top = {
    'linear': Linear
}
