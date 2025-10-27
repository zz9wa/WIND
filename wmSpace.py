import torch
import torch.nn as nn
import torch.nn.functional as F

class wmDiscriminator(nn.Module):
    def __init__(self, wm_dim):
        super(wmDiscriminator, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(wm_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class BitStringMapper(nn.Module):
    def __init__(self,args):
        super(BitStringMapper, self).__init__()


        self.mapping_matrix = nn.Parameter(torch.randn(args.feature_dim, args.wm_dim).to(args.gpu))


        self.sigmoid = nn.Sigmoid()

    def forward(self, x, args):

        x = torch.matmul(x, self.mapping_matrix)
        x = self.sigmoid(x)
        return x

class WatermarkEncoder(nn.Module):
    def __init__(self, feature_dim, watermark_dim):
        super(WatermarkEncoder, self).__init__()
        self.fc_embed = nn.Linear(feature_dim, watermark_dim)
        self.fc_gen = nn.Linear(feature_dim + watermark_dim, feature_dim)

    def forward(self, text_features, args,watermark=None):
        self.to(args.gpu)
        if watermark is not None:

            print("text_features",text_features)
            print("shape",text_features.shape)
            embedded_features = torch.cat((text_features.to(args.gpu), watermark.to(args.gpu)), dim=1)
            return self.fc_gen(embedded_features)
        else:
            return text_features

class wmModel(nn.Module):
    def __init__(self, feature_dim, watermark_dim):
        super(wmModel, self).__init__()
        self.watermark_encoder = WatermarkEncoder(feature_dim, watermark_dim)
        self.discriminator = BitStringMapper(feature_dim)

    def forward(self, text_features, args, watermark=None):
        watermarked_features = self.watermark_encoder(text_features, args,watermark)

        return self.discriminator(watermarked_features,args)