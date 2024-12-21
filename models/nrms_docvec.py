import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import torch.nn.functional as F

from models.layers import AttLayer2


class NewsEncoder(nn.Module):
    def __init__(self, hparams, seed=None):
        super(NewsEncoder, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        DOCUMENT_VECTOR_DIM = hparams.title_size
        OUTPUT_DIM = hparams.head_num * hparams.head_dim
        self.layers = nn.ModuleList()
        for units in hparams.newsencoder_units_per_layer:
            self.layers.append(
                nn.Sequential(
                    nn.Linear(DOCUMENT_VECTOR_DIM, units),
                    nn.ReLU(),
                    nn.BatchNorm1d(units),
                    nn.Dropout(hparams.dropout)
                )
            )
            DOCUMENT_VECTOR_DIM = units

        self.output_layer = nn.Linear(DOCUMENT_VECTOR_DIM, OUTPUT_DIM)



    def forward(self, x):
        # x: (B, title_size)
        for layer in self.layers:
            x = layer(x)
        # x = self.output_layer(x)  # (B, output_dim)
        # print(f"Input to NewsEncoder: {x.shape}")
        x = self.output_layer(x)
        # print(f"Output from NewsEncoder: {x.shape}")
        x = F.relu(x)
        return x

class UserEncoder(nn.Module):
    def __init__(self, hparams, newsencoder, seed=None):
        super(UserEncoder, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self.newsencoder = newsencoder
        # Initialize MultiheadAttention
        self.attention = nn.MultiheadAttention(
            embed_dim=hparams.head_num * hparams.head_dim,
            num_heads=hparams.head_num,
            batch_first=True
        )       
        self.att_layer = AttLayer2(hparams.attention_hidden_dim, seed=seed)

        self.history_size = hparams.history_size

    def forward(self, his_input_title):
        B, H, F = his_input_title.shape
        # print(f"Input to UserEncoder: {his_input_title.shape}")
        x = his_input_title.view(B * H, F)
        # print(f"After flatten: {x.shape}")
        x = self.newsencoder(x)
        # print(f"After newsencoder: {x.shape}")
        x = x.view(B, H, -1)
        # print(f"After reshape: {x.shape}")
        
        # MultiheadAttention expects inputs in (batch_size, seq_len, embed_dim) or similar
        attn_output, _ = self.attention(x, x, x)  # Query, Key, Value all as x
        # print(f"After MultiheadAttention: {attn_output.shape}")

        # Apply the AttLayer2 to get the user presentation
        user_present = self.att_layer(attn_output)
        # print(f"Final user_present: {user_present.shape}")

        return user_present

class NRMSModel(nn.Module):
    def __init__(self, hparams, userencoder, newsencoder):
        super(NRMSModel, self).__init__()
        self.userencoder = userencoder
        self.newsencoder = newsencoder

    def forward(self, his_input_title, pred_input_title):
        B, N, F = pred_input_title.shape

        user_present = self.userencoder(his_input_title)  # (B, output_dim)
        pred_input_reshaped = pred_input_title.view(B*N, F)
        news_present = self.newsencoder(pred_input_reshaped)
        news_present = news_present.view(B, N, -1)  # (B,N,output_dim)

        # torch.bmm 
        user_present = user_present.unsqueeze(2)  # [B, output_dim, 1]
        preds = torch.bmm(news_present, user_present).squeeze(2)  # [B, N]
        
        if not isinstance(preds, torch.Tensor):
            preds = torch.tensor(preds, dtype=torch.float32, device=his_input_title.device)
            
        preds = torch.nn.functional.softmax(preds, dim=-1)  # [B, N]
        return preds

class NRMScorer(nn.Module):
    def __init__(self, hparams, userencoder, newsencoder):
        super(NRMScorer, self).__init__()
        self.userencoder = userencoder
        self.newsencoder = newsencoder

    def forward(self, his_input_title, pred_input_title_one):
        # pred_input_title_one: (B,1,title_size)
        B, _, F = pred_input_title_one.shape
        pred_input_title_one_reshaped = pred_input_title_one.view(B, F)

        user_present = self.userencoder(his_input_title)
        news_present_one = self.newsencoder(pred_input_title_one_reshaped)

        pred_one = (news_present_one * user_present).sum(dim=-1)  # (B)
        pred_one = torch.sigmoid(pred_one) 
        return pred_one

class NRMSDocVec:
    def __init__(self, hparams: dict, seed: int = None):
        self.hparams = hparams
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)


        self.model, self.scorer = self._build_graph()

        self.criterion = self._get_loss(self.hparams.loss)
        self.optimizer = self._get_opt(self.hparams.optimizer, self.hparams.learning_rate, self.model.parameters())

    def _get_loss(self, loss: str):
        if loss == "cross_entropy_loss":

            return nn.CrossEntropyLoss()
        elif loss == "log_loss":

            return nn.BCEWithLogitsLoss()  
        else:
            raise ValueError(f"this loss not defined {loss}")

    def _get_opt(self, optimizer: str, lr: float, params):
        if optimizer == "adam":
            return torch.optim.Adam(params, lr=lr)
        else:
            raise ValueError(f"this optimizer not defined {optimizer}")

    def _build_graph(self):

        newsencoder = self._build_newsencoder(self.hparams.newsencoder_units_per_layer)

        userencoder = self._build_userencoder(newsencoder)


        model = NRMSModel(self.hparams, userencoder, newsencoder)
        scorer = NRMScorer(self.hparams, userencoder, newsencoder)
        self.userencoder = userencoder
        self.newsencoder = newsencoder
        return model, scorer

    def _build_userencoder(self, titleencoder):
        userencoder = UserEncoder(self.hparams, titleencoder, seed=self.seed)
        return userencoder

    def _build_newsencoder(self, units_per_layer):
        newsencoder = NewsEncoder(self.hparams, seed=self.seed)
        return newsencoder
