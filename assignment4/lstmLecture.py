# lstm_lecture.py

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.spatial.distance


class RNNOverWords(nn.Module):
    def __init__(self, dict_size, input_size, hidden_size, dropout, rnn_type='lstm'):
        super(RNNOverWords, self).__init__()
        self.word_embedding = nn.Embedding(dict_size, input_size)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, dropout=dropout)
        self.init_weight()

    def init_weight(self):
        # This is a randomly initialized RNN.
        # Bounds from https://openreview.net/pdf?id=BkgPajAcY7
        # Note: this is to make a random LSTM; these are *NOT* necessarily good weights for initializing learning!
        nn.init.uniform_(self.rnn.weight_hh_l0, a=-1.0/np.sqrt(self.hidden_size), b=1.0/np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.weight_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_hh_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))
        nn.init.uniform_(self.rnn.bias_ih_l0, a=-1.0 / np.sqrt(self.hidden_size),
                        b=1.0 / np.sqrt(self.hidden_size))

    def forward(self, input):
        embedded_input = self.word_embedding(input)
        # RNN expects a batch
        embedded_input = embedded_input.unsqueeze(1)
        # Note: the hidden state and cell state are 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        # So we need to unsqueeze to add these 1-dims.
        init_state = (torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float(),
                      torch.from_numpy(np.zeros(self.hidden_size)).unsqueeze(0).unsqueeze(1).float())
        output, (hidden_state, cell_state) = self.rnn(embedded_input, init_state)
        # Note: hidden_state is a 1x1xdim tensor: num layers * num directions x batch_size x dimensionality
        return output, hidden_state, cell_state


# The biases for LSTMs are all encoded into a single array, with 4 biases packed into one array bias_ih_l0.
# The first 1/4 of the values correspond to the input gate bias, and the second 1/4 of the values correspond
# to the forget gate bias.
def set_input_bias(rnn_module, bias):
    rnn_bias = rnn_module.rnn.bias_ih_l0
    start = 0
    end = len(rnn_bias) // 4
    rnn_bias.data[start:end].fill_(bias)


def set_forget_bias(rnn_module, bias):
    rnn_bias = rnn_module.rnn.bias_ih_l0
    start = len(rnn_bias) // 4
    end = len(rnn_bias) // 2
    rnn_bias.data[start:end].fill_(bias)


def check_distance(rnn_module, inps, plot=False):
    hiddens = []
    for inp in inps:
        output, hidden_state, cell_state = rnn_module.forward(inp)
        hiddens.append(cell_state)
    # Use this if you want to plot the results
    if plot:
        for (hidden, inp) in zip(hiddens, inps):
            hidden_vec = np.array(hidden.data[0][0])
            plt.plot(hidden_vec[0], hidden_vec[1], color='k', linestyle='', marker=".")
            plt.text(hidden_vec[0]+0.001, hidden_vec[1], repr(np.asarray(inp)))
        plt.title('SGD on quadratic')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    # Compare embeddings after the 0th sequence to embeddings after the other sequences
    for j in range(1, len(hiddens)):
        h0_vec = np.array(hiddens[0].data[0][0])
        hj_vec = np.array(hiddens[j].data[0][0])
        print("Euclidean distance between cell state after " + repr(inps[0]) + " and cell state after " +
              repr(inps[j]) + ": %.3f (cell state magnitudes = %.3f and %.3f)" %
              (scipy.spatial.distance.euclidean(h0_vec, hj_vec), np.sqrt(np.dot(h0_vec, h0_vec)), np.sqrt(np.dot(hj_vec, hj_vec))))


if __name__=="__main__":
    rnn_module = RNNOverWords(dict_size=5, input_size=5, hidden_size=2, dropout=0.)
    # Change last, second-to-last, and third-to-last positions
    inps = [torch.from_numpy(np.array([0, 1, 2])),
            torch.from_numpy(np.array([0, 1, 3])),
            torch.from_numpy(np.array([0, 4, 2])),
            torch.from_numpy(np.array([1, 1, 2]))]
    print("Standard LSTM with random parameters")
    check_distance(rnn_module, inps)
    print("======")
    print("Forget gate bias set high -- LSTM remembers a lot")
    set_forget_bias(rnn_module, 10.)
    check_distance(rnn_module, inps)
    print("======")
    print("Forget gate bias set low -- LSTM remembers little")
    set_forget_bias(rnn_module, -10.)
    check_distance(rnn_module, inps)
    print("======")
    print("Input gate bias set high -- LSTM is very sensitive to its inputs")
    set_forget_bias(rnn_module, 0.)
    set_input_bias(rnn_module, 10.)
    check_distance(rnn_module, inps)
    print("======")
    print("Input gate bias set low -- LSTM inputs don't affect it much")
    set_forget_bias(rnn_module, 0.)
    set_input_bias(rnn_module, -10.)
    check_distance(rnn_module, inps)


