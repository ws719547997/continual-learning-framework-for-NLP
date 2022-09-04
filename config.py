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
    """
        CLUE:
        clue.csl clue.afqmc clue.cluewsc2020 clue.cmnli clue.iflytek clue.tnews

        JD:
        jd21.褪黑素 jd21.维生素 jd21.无线耳机 jd21.蛋白粉 jd21.游戏机 jd21.电视 jd21.MacBook
        jd21.洗面奶 jd21.智能手表 jd21.吹风机 jd21.小米手机 jd21.红米手机 jd21.护肤品 jd21.电动牙刷
        jd21.iPhone jd21.海鲜 jd21.酒 jd21.平板电脑 jd21.修复霜 jd21.运动鞋 jd21.智能手环

        STOCK:
        stock.airline stock.car stock.communication stock.energy stock.finance stock.manufacture
        stock.medical stock.Real_estate stock.tech stock.traffic stock.wine

        AMZ:
        amz20.Sandal amz20.Magazine_Subscriptions amz20.RiceCooker amz20.Flashlight amz20.Jewelry
        amz20.CableModem amz20.GraphicsCard amz20.GPS amz20.Projector amz20.Keyboard amz20.Video_Games
        amz20.AlarmClock amz20.HomeTheaterSystem amz20.Vacuum amz20.Gloves amz20.Baby amz20.Bag
        amz20.Movies_TV amz20.Dumbbell amz20.Headphone

        SNAP:
        snap10k.Automotive_5 snap10k.Electronics_5 snap10k.Industrial_and_Scientific_5 snap10k.Kindle_Store_5
        snap10k.Cell_Phones_and_Accessories_5 snap10k.Musical_Instruments_5 snap10k.Office_Products_5
        snap10k.Patio_Lawn_and_Garden_5 snap10k.Sports_and_Outdoors_5 snap10k.Luxury_Beauty_5
        snap10k.Grocery_and_Gourmet_Food_5 snap10k.Digital_Music_5 snap10k.Tools_and_Home_Improvement_5
        snap10k.Pet_Supplies_5 snap10k.Prime_Pantry_5 snap10k.Toys_and_Games_5 snap10k.Movies_and_TV_5
        snap10k.Home_and_Kitchen_5 snap10k.Arts_Crafts_and_Sewing_5 snap10k.Video_Games_5 snap10k.CDs_and_Vinyl_5
    """

    parser.add_argument("--task_list",
                        default='tasklist/jd21.txt',
                        type=str,
                        help='input task list')
    parser.add_argument("--model_name", default='model/bert-base-uncased', type=str,
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
