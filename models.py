import torch
from torch import nn
import random

class Encoder(nn.Module):
    def __init__(self, embd_matrix:torch.Tensor, pretrained:bool, lstm_hidden_size, lstm_layers=1, dropout_probability=0.1):
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.input_size = embd_matrix.size(-1)
        self.embd_layer = nn.Embedding.from_pretrained(embd_matrix, freeze=False) if pretrained else nn.Embedding(embd_matrix.size(0), embd_matrix.size(1))

        self.dropout = nn.Dropout(dropout_probability)
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.lstm_layers, dropout=dropout_probability, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_size * 2, lstm_hidden_size)

    def forward(self, x):
        embds = self.dropout(self.embd_layer(x))
        output, hidden = self.gru(embds)
        # hidden [-2, :, : ] is the last of the forwards RNN
        # hidden [-1, :, : ] is the last of the backwards RNN
        # all_hidden = hidden
        hidden = torch.cat([hidden[-2,:,:], hidden[-1,:,:]], dim=1)
        hidden = torch.tanh(self.fc(hidden))

        return output, hidden

########################--------------------##################################
class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear((encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim)
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [src length, batch size, encoder hidden dim * 2]
        batch_size = encoder_outputs.shape[0]
        src_length = encoder_outputs.shape[1]
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        # hidden = [batch size, src length, decoder hidden dim]
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        pre_energy = torch.cat((hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn_fc(pre_energy))
        
        # energy = [batch size, src length, decoder hidden dim]
        attention = self.v_fc(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1)

########################--------------------##################################
class Decoder(nn.Module):
    def __init__(self, embd_matrix:torch.Tensor, pretrained:bool, attention:Attention, lstm_hidden_size, lstm_layers=1, dropout_probability=0):
        super().__init__()
        self.hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.input_size = embd_matrix.size(-1)
        self.embd_layer = nn.Embedding.from_pretrained(embd_matrix, freeze=False) if pretrained else nn.Embedding(embd_matrix.size(0), embd_matrix.size(1))
        self.attention = attention

        self.dropout = nn.Dropout(dropout_probability)
        self.gru = nn.GRU((lstm_hidden_size * 2) + self.input_size, self.hidden_size, self.lstm_layers, dropout=dropout_probability, batch_first=True)
        self.fc_out = nn.Linear(self.hidden_size, embd_matrix.size(0))
        
    def forward(self, x, hidden_t_1, encoder_outputs):
        embds = self.dropout(self.embd_layer(x))
        a = self.attention(hidden_t_1, encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embds, weighted), dim=2)

        output, hidden_t = self.gru(rnn_input, hidden_t_1.unsqueeze(0))
        # embds = embds.squeeze(1)
        output = output.squeeze(1)
        # weighted = weighted.squeeze(1)
        # all_in_one = torch.cat((output, weighted, embds), dim=1)
        # prediction = self.fc_out(all_in_one)
        prediction = self.fc_out(output)
        
        return prediction, hidden_t.squeeze(0), a.squeeze(1)


########################--------------------##################################
class Seq2seq_with_attention(nn.Module):
    def __init__(self, encoder:Encoder, decoder:Decoder):
        super(Seq2seq_with_attention, self).__init__()

        self.decoder_vocab_size = decoder.embd_layer.weight.size(0)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):

        batch_size, seq_len = target.size()

        total_outputs = torch.zeros(batch_size, seq_len, self.decoder_vocab_size, device=source.device)

        encoder_outputs, hidden = self.encoder(source)

        x = target[:, [0]]
        for step in range(1, seq_len):
            logits, hidden, _ = self.decoder(x, hidden, encoder_outputs)
            
            total_outputs[:, step] = logits
            top1 = logits.argmax(1, keepdim=True)
            x = target[:, [step]] if teacher_force_ratio > random.random() else top1

        return total_outputs

    @torch.no_grad
    def translate(self, source:torch.Tensor, max_tries=50):
        output = [2] ## <SOS> token

        targets = torch.randint(0,1,(source.size(0), max_tries)).to(device=source.device)
        targets_hat = self.forward(source, targets, 0.0)
        targets_hat = targets_hat.argmax(-1).squeeze(0)
        for token in targets_hat[1:]:
            output.append(token.item())
            if token == 3:
                break

        return output