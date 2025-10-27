import torch
from torch import nn
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,BertModel, BertTokenizer
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from utils import (load_data,process_sen_pairs,create_loss_fn)
#from simcse import SimCSE
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn import functional as F
from scipy.spatial.distance import cosine
from termcolor import colored
import datetime
from sklearn.linear_model import Ridge
from wmSpace import BitStringMapper, WatermarkEncoder,wmModel
from torch.optim.lr_scheduler import LambdaLR


class d(nn.Module):
    def __init__(self,args):
        super(d, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 2)  # 两类任务

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class RidgeClassifier:
    def __init__(self):
        self.ridge = Ridge(alpha=1.0)

    def fit(self, X, y):
        self.ridge.fit(X, y)

    def predict(self, X):
        return self.ridge.predict(X)



class ContrastiveTrainer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = args.gpu
        #self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        if args.encoder=='sup-simcse-bert-base-uncased':
            self.tokenizer =AutoTokenizer.from_pretrained(args.encoder_path)#BertTokenizer.from_pretrained(model_name)#
            self.model =AutoModel.from_pretrained(args.encoder_path).to(self.device)#BertModel.from_pretrained(model_name).to(self.device)
        elif args.encoder=='bert-base-uncased':
            self.tokenizer =BertTokenizer.from_pretrained(args.encoder_path)#
            self.model =BertModel.from_pretrained(args.encoder_path).to(self.device)
        elif args.encoder == 'robota':
            self.tokenizer = RobertaTokenizer.from_pretrained(args.encoder_path)  #
            self.model = RobertaModel.from_pretrained(args.encoder_path).to(self.device)
        self.batch_size = args.batch_size


        #self.watermark_encoder = WatermarkEncoder(args.feature_dim, args.wm_dim)
        self.wmMatrix = BitStringMapper(args)

        self.models = nn.ModuleDict({
            'encoder': self.model,
            'watermark_encoder': self.wmMatrix
        })



    def forward(self, texts,args,watermark=None):

        encodings = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        encodings = {key: val.to(self.device) for key, val in encodings.items()}

        outputs = self.model(**encodings, output_hidden_states=True, return_dict=True).pooler_output

        if watermark is not None:

            #watermarked_features = self.watermark_encoder(outputs, args, watermark)

            watermark_output = self.wmMatrix(watermark, args)
            return outputs, watermark, watermark_output
        else:
            return outputs

    def get_optimizer(self, epoch, args):

        if args.now_epoch < 4:

            encoder_lr = 5e-5
            watermark_encoder_lr = 1e-5
        else:

            encoder_lr = 1e-7
            watermark_encoder_lr = 1e-5


        optimizer_grouped_parameters = [
            {
                'params': self.models['encoder'].parameters(),
                'lr': encoder_lr
            },
            {
                'params': self.models['watermark_encoder'].parameters(),
                'lr': watermark_encoder_lr
            }
        ]


        #print("lr:", optimizer_grouped_parameters[0]['lr'])
        optimizer = AdamW(optimizer_grouped_parameters)
        return optimizer
def create_optimizer(model,args):

    optimizer_params = [
        {'params': model.model.parameters(), 'lr': 2e-5},
        {'params': model.wmMatrix.parameters(), 'lr': 5e-4}
    ]
    optimizer = AdamW(optimizer_params, weight_decay=1e-5 )

    return optimizer









