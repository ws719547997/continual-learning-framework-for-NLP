from approaches.bert_nn import Appr as bert_nn

appr_dict = {
    'bert_last_ncl': bert_nn
}


def build_approaches(model, args, device):
    return appr_dict['bert_last_ncl'](model, args, device)
