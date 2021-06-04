import torch
from torch import nn
import torch.nn.functional as F
from downstream_models.base_model import DownstreamBaseModel, DownstreamBaseTrainer



class LSTMModel(DownstreamBaseModel):
    def __init__(self, params=None, vocab_size=36, hidden_size=50, num_layers=1,
                 dropout=0.0, embedding_size=40000, bidirectional=True, *args, **kwargs):
        super(LSTMModel, self).__init__(params)
        self.embedder1 = nn.Embedding(num_embeddings=embedding_size, embedding_dim=vocab_size)
        self.embedder2 = nn.Embedding(num_embeddings=embedding_size, embedding_dim=vocab_size)
        self.embedder3 = nn.Embedding(num_embeddings=embedding_size, embedding_dim=vocab_size)

        # for the encoder network:
        self.embedderSmall1 = nn.Embedding(num_embeddings=embedding_size // 2, embedding_dim=vocab_size // 3)
        self.embedderSmall2 = nn.Embedding(num_embeddings=embedding_size // 2, embedding_dim=vocab_size // 3)
        self.embedderSmall3 = nn.Embedding(num_embeddings=embedding_size // 2, embedding_dim=vocab_size // 3)
        self.num_layers = num_layers

        self.biLSTM_left = nn.LSTM(vocab_size, hidden_size, num_layers,
                                   dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.biLSTM_bet = nn.LSTM(vocab_size, hidden_size, num_layers,
                                  dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.biLSTM_right = nn.LSTM(vocab_size, hidden_size, num_layers,
                                    dropout=dropout, bidirectional=bidirectional, batch_first=True)

        if bidirectional:
            self.num_layers *= 2
        flattened = 3 * hidden_size
        self.out_size = 1 if params is None else params['out_dim']
        self.decoder1 = nn.Linear(in_features=flattened, out_features=64)
        self.decoder2 = nn.Linear(in_features=64, out_features=32)
        self.readout = nn.Linear(in_features=32, out_features=self.out_size)
        self.hidden_size = hidden_size
        self.weight_size = (num_layers, vocab_size, hidden_size)

    def init_hidden(self, batch_size, device='cuda'):
        return (torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_size)).to(device),
                torch.autograd.Variable(torch.randn(self.num_layers, batch_size, self.hidden_size)).to(device))

    def embed_features(self, x, device='cuda', small=False):
        left_ph, between_ph, right_ph = x[0].to(device), x[1].to(device), x[2].to(device)
        if not small:
            return self.embedder1(left_ph), self.embedder2(between_ph), self.embedder3(right_ph)
        else:
            return self.embedderSmall1(left_ph), self.embedderSmall2(between_ph), self.embedderSmall3(right_ph)

    def get_encoder_features(self, x, device='cuda'):
        feats_for_enc = self.embed_features(x, small=True, device=device)
        feats_for_enc = torch.cat([emb.reshape((x[0].shape[0], -1)) for emb in feats_for_enc], dim=1)
        # feats_for_enc = self.forward(x, device=device, readout=False).detach()
        return feats_for_enc

    def forward(self, x, device='cuda', readout=True):
        left_ph, between_ph, right_ph = self.embed_features(x, device=device)
        hidden0 = self.init_hidden(left_ph.shape[0])
        hidden1 = self.init_hidden(between_ph.shape[0])
        hidden2 = self.init_hidden(right_ph.shape[0])

        l_output, (lhidden, lcell) = self.biLSTM_left(left_ph, hidden0)
        b_output, (bhidden, bcell) = self.biLSTM_bet(between_ph, hidden1)
        r_output, (rhidden, rcell) = self.biLSTM_right(right_ph, hidden2)

        lstm_output = torch.cat((lhidden[-1], bhidden[-1], rhidden[-1]), dim=1)

        if readout:
            decoded = F.relu(self.decoder1(lstm_output))
            decoded = F.relu(self.decoder2(decoded))
            logits = self.readout(decoded)
            return logits.squeeze(1)
        else:
            return lstm_output

    def __str__(self):
        return 'LSTM'


class LSTM_Trainer(DownstreamBaseTrainer):
    def __init__(
            self, downstream_params, name='LSTM', seed=None, verbose=False,
            model_dir="out/LSTM", notebook_mode=False, model=None
    ):
        super().__init__(downstream_params, name=name, seed=seed, verbose=verbose,
                         model_dir=model_dir, notebook_mode=notebook_mode, model=model)
        self.model_class = LSTMModel
        self.name = name
