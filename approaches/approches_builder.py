from approaches.bert_nn import Appr as bert_nn


def get_approach(args):
    if args.approach == 'bert_last_ncl':
        return bert_nn


def build_approaches(model, taskmaneger, args, device):
    return get_approach(args)(model, taskmaneger, args, device)
