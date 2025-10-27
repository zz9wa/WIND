import argparse
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description=" Args ")

    parser.add_argument("--train_data", type=str,
                        default="",
                        help="path to dataset")
    parser.add_argument("--test_data", type=str,
                        default="data/reuters.json",
                        help="path to dataset")
    parser.add_argument("--save_path", type=str,
                        default="",
                        help="model path")
    parser.add_argument("--encoder", type=str,
                        default="sup-simcse-bert-base-uncased",
                        help="encoder model")
    parser.add_argument("--encoder_path", type=str,
                        default="sup-simcse-bert-base-uncased",
                        help="encoder model path")
    parser.add_argument("--log_path", type=str,
                        default="",
                        help="train log path")
    parser.add_argument("--test_model_path", type=str,
                        default="",
                        help="tested model path")

    parser.add_argument("--num", type=int, default=5,
                        help="number_sentences")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=2,
                        help="epochs")
    parser.add_argument("--seed", type=int, default=233,
                        help="random seed")
    parser.add_argument("--now_epoch", type=int, default=0,
                        help="now epoch")

    parser.add_argument("--feature_dim", type=int, default=768,
                        help="Feature dimension")
    parser.add_argument("--wm_dim", type=int, default=10,
                        help="Watermark dimension")
    parser.add_argument("--wm_thres", type=float, default=0.5,
                        help="Watermark thresolds")


    parser.add_argument("--gpu", type=int, default=0,
                        help="device:gpu")
    parser.add_argument("--mode", type=str, default='train',
                        help="device:gpu")
    parser.add_argument("--set", type=str, default=' ',
                        help="exp setting")
    parser.add_argument("--patiance", type=int, default=3,
                        help="patiance for training")




    return parser.parse_args()