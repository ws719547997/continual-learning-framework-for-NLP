def build_method(args):
    name = args.approach
    if name == 'bert_last_ncl':
        from approaches.bert_nn import Appr
        from models.bert_fixed_cnn import Net
    return Appr, Net