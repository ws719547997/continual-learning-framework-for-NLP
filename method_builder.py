def build_method(args):
    name = args.approach
    if name == 'last':
        from approaches.bert_nn import Appr
        from models.bert_textcnn import Net
    return Appr, Net