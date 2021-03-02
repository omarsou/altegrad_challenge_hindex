import torch.nn as nn
import numpy as np
import torch
from src.models_block import SpecialEmbeddings, TransformerEncoder


class HindexModel(nn.Module):
    def __init__(self, encoder, decoder, paper_embed, author_embed, features_embed, criterion, dim_decoder, dropout):
        super(HindexModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.paper_embed = paper_embed
        self.author_embed = author_embed
        self.features_embed = features_embed
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(dim_decoder, 1)
        self.loss = criterion

    def forward(self, batch):
        features_embed = self.features_embed(batch.features, None)
        author_embed = self.author_embed(batch.autemb, batch.autemb_mask)
        paper_embed = self.paper_embed(batch.papemb, batch.papemb_mask)
        input_encoder = paper_embed + features_embed + author_embed
        output_encoded, _ = self.encoder(input_encoder, batch.papemb_mask)
        out = output_encoded.flatten(start_dim=1)
        out = self.dropout(self.decoder(out))
        out = self.activation(out)
        return self.fc(out)

    def get_metrics_for_batch(self, batch):
        output = self.forward(batch)
        loss = self.loss(output, batch.target)
        rmse = np.sqrt(loss.detach().cpu().item())
        mae = torch.mean(torch.abs(output - batch.target)).detach().cpu().item()
        return loss, rmse, mae


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)


def build_model(cfg):
    paper_embed = SpecialEmbeddings(embedding_dim=512, input_size=cfg.paper_dim, num_heads=cfg.num_heads, mask_on=True)
    author_embed = SpecialEmbeddings(embedding_dim=512, input_size=cfg.author_dim, num_heads=cfg.num_heads,
                                     mask_on=False)
    features_embed = SpecialEmbeddings(embedding_dim=512, input_size=4, num_heads=cfg.num_heads, mask_on=False)
    encoder = TransformerEncoder(num_layers=cfg.num_layers, num_heads=cfg.num_heads)
    decoder = nn.Linear(512 * 10, 512)
    criterion = nn.MSELoss()
    model = HindexModel(encoder, decoder, paper_embed, author_embed, features_embed, criterion, 512, cfg.dropout)
    model.apply(init_weights)
    if cfg.use_cuda:
        model.cuda()
    return model
