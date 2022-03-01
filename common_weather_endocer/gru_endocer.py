from  torch import nn
from util import *

class WeatherEncoder(nn.Module):
    def __init__(self, config):
        super(WeatherEncoder, self).__init__()
        self.out_size = config['weather.out_size']
        self.hidden_size = config['weather.hidden_size']
        self.input_size = config['weather.input_size']
        self.layers = config['weather.num_layers']
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.layers,
                          # dropout=0.5
                          )

        self.fc_out = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.out_size)
        )

    def forward(self, x):
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        #x = x.permute(2, 0, 1)  # seq_len,batch,input_dim
        batch,seq_len,input_dim = x.shape
        h = init_rnn_hidden(batch=batch, hidden_size=self.hidden_size, num_layers=self.layers)
        _, h = self.gru(x.permute(1,0,2), h)
        out = self.fc_out(h[-1, :, :])
        return out
