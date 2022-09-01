import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser = basic_args(parser)
    parser = process_args(parser)
    parser = train_args(parser)
    parser = eval_args(parser)
    parser = model_args(parser)

    args = parser.parse_args()
    return args


def basic_args(parser):
    '''
        CLUE:
        clue.csl clue.afqmc clue.cluewsc2020 clue.cmnli clue.iflytek clue.tnews

        JD:
        jd21.iPhone jd21.修复霜

    '''
    parser.add_argument("--task_list", default='clue.csl clue.afqmc clue.cluewsc2020 clue.cmnli clue.iflytek clue.tnews',
                        type=str,
                        help='input task list')
    parser.add_argument("--model_name", default='model/bert-base-chinese', type=str,
                        help='bert模型存放的目录')
    return parser


def process_args(parser):
    return parser


def train_args(parser):
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    return parser


def eval_args(parser):
    return parser


def model_args(parser):
    return parser
