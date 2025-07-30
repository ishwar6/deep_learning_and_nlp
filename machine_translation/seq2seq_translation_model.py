import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        src_len = src.shape[0]
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        outputs = torch.zeros(trg_len, batch_size, self.decoder.output_dim)

        encoder_outputs, hidden = self.encoder(src)
        input = trg[0, :]  

        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs)
            outputs[t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            input = trg[t] if teacher_force else output.argmax(1)

        return outputs

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden

def main():
    SRC = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)
    TRG = Field(tokenize='spacy', init_token='<sos>', eos_token='<eos>', lower=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    input_dim = len(SRC.vocab)
    output_dim = len(TRG.vocab)
    enc_emb_dim = 256
    dec_emb_dim = 256
    enc_hidden_dim = 512
    dec_hidden_dim = 512
    n_layers = 2
    dropout = 0.5
    encoder = Encoder(input_dim, enc_emb_dim, enc_hidden_dim, n_layers, dropout)
    decoder = Decoder(output_dim, dec_emb_dim, dec_hidden_dim, n_layers, dropout)
    model = Seq2Seq(encoder, decoder)
    print(model)

if __name__ == '__main__':
    main()