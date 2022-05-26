import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

class YOURLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(YOURLSTM, self).__init__()

        self.LSTM_cell1 = nn.LSTMCell(input_size, hidden_size) 
        self.LSTM_cell2 = nn.LSTMCell(input_size, hidden_size)
        self.LSTM_cell3 = nn.LSTMCell(input_size, hidden_size)
        self.LSTM_cell4 = nn.LSTMCell(input_size, hidden_size)

    def forward(self, x, state):
        h_0, c_0 = state[0].clone(), state[1].clone() # 각각 (4, 128, 512)
        cur_batch = x.shape[0]

        state_0 = self.LSTM_cell1(x, (h_0[0][:cur_batch].clone(), c_0[0][:cur_batch].clone())) # e.g. (125, 512)
        state_1 = self.LSTM_cell2(state_0[0], (h_0[1][:cur_batch].clone(), c_0[1][:cur_batch].clone()))
        state_2 = self.LSTM_cell3(state_1[0], (h_0[2][:cur_batch].clone(), c_0[2][:cur_batch].clone()))
        state_3 = self.LSTM_cell4(state_2[0], (h_0[3][:cur_batch].clone(), c_0[3][:cur_batch].clone()))

        out_h = torch.stack([state_0[0].clone(), state_1[0].clone(), state_2[0].clone(), state_3[0].clone()], dim=0).squeeze() # (4, 128, 512)
        out_c = torch.stack([state_0[1].clone(), state_1[1].clone(), state_2[1].clone(), state_3[1].clone()], dim=0).squeeze()

        return (out_h.clone(), out_c.clone())

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=4, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        """ TO DO: Implement your LSTM """
        self.rnn = YOURLSTM(hidden_size, hidden_size, num_layers, batch_first=True)
	
    def forward(self, x, state): # x: (128, 20), state: tuple((4,128,512), (4,128,512))
        """ TO DO: feed the unpacked input x to Encoder """
        inputs_length = torch.sum(torch.where(x > 0, True, False), dim=1) # 각 sentence별 pad가 아닌 길이 tensor(128,)
        x = self.embedding(x) # x: (128, 20, 512)
        packed = pack(x, inputs_length.tolist(), batch_first=True, enforce_sorted=False)

        max_len = len(packed[1]) # 20

        packed_data = packed[0].clone().detach()
        packed_batch_sizes = packed[1].clone().detach()
        tmp_h, tmp_c = torch.zeros(state[0].shape[1], state[0].shape[2]), torch.zeros(state[1].shape[1], state[1].shape[2]) # 각각 (128, 512)

        start = 0
        for i in range(max_len):
            cur_batch = packed_batch_sizes[i].clone().item() # e.g. 125
            cur_input = packed_data[start:start+cur_batch]
           
            tmp_state = self.rnn(cur_input, state) # state: e.g. tuple (4, 125, 512)
            state[0][:,:cur_batch,:] = tmp_state[0].clone()
            state[1][:,:cur_batch,:] = tmp_state[1].clone()
            packed[0][start:start+cur_batch] = tmp_state[0][3].clone()
            start = start + cur_batch

            if i == 0: continue # In 1st step(idx: 0), every sample in this mini-batch will have token.
            elif i == 19:  # In 20th token(idx: 19 when max_len: 20), this if final step so store every tokens.
                tmp_h[:cur_batch], tmp_c[:cur_batch] = tmp_state[0][3].clone(), tmp_state[1][3].clone()
            else:  # 이전 len~현재 len까지 다 저장.
                tmp_h[cur_batch-1:packed_batch_sizes[i-1]] = tmp_state[0][3][cur_batch-1:packed_batch_sizes[i-1]].clone()
                tmp_c[cur_batch - 1:packed_batch_sizes[i - 1]] = tmp_state[1][3][cur_batch - 1:packed_batch_sizes[i - 1]].clone()

        output, outputs_length = unpack(packed, batch_first=True, total_length=x.shape[1])

        return output, state

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=4, ar=None, tf=None):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
		
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        """ TO DO: Implement your LSTM """
        self.rnn = YOURLSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        self.classifier = nn.Sequential(
			nn.Linear(hidden_size, vocab_size),
			nn.LogSoftmax(dim=-1)
		)

        self.autoregressive = ar # True / False
        self.teacher_forcing = tf # True / False
	
    def forward(self, max_len, tgt, state, device):
        """ TO DO: feed the input x to Decoder """
        eos = torch.tensor([2]).repeat(state[0].shape[1]).to(device) # (128,)
        x = self.embedding(eos.clone()) # (128, 512)

        output = []
        # output = torch.zeros(max_len, tgt.shape[0], self.vocab_size)
        if self.autoregressive: # if autoregressive
            if not self.teacher_forcing:
                print("Autoregressive O, Teacher Forcing X, Attention X")
                for i in range(max_len):
                    if i == 0:
                        state = self.rnn(x.clone(), state) # x: (128, 512), state: tuple ((4, 128, 512), (4, 128, 512))
                    else:
                        prev_out = torch.argmax(output[i-1], dim=1)
                        inp = self.embedding(prev_out.clone())
                        state = self.rnn(inp.clone(), state)
                    out = self.classifier(state[0][3].clone())
                    output.append(out.clone())
            else: # implement ar teacher forcing
                print("Autoregressive O, Teacher Forcing O, Attention X")
                # tgt: (128, 20)

                for i in range(max_len):
                    if i == 0:
                        state = self.rnn(x.clone(), state)  # x: (128, 512), state: tuple ((4, 128, 512), (4, 128, 512))
                    else:
                        inp = self.embedding(tgt[:, i - 1].clone())
                        state = self.rnn(inp.clone(), state)
                    out = self.classifier(state[0][3].clone())
                    output.append(out.clone())

        else: # if non-autoregressive
            if not self.teacher_forcing:
                print("Autoregressive X, Teacher Forcing X, Attention X")
                for i in range(max_len):
                    if i == 0:
                        state = self.rnn(x.clone(), state)
                    else:
                        prev_out = torch.argmax(x, dim=1) # input eos as input every time
                        inp = self.embedding(prev_out.clone())
                        state = self.rnn(inp.clone(), state)
                    out = self.classifier(state[0][3].clone())
                    output.append(out.clone())

            else: # implement non-autoregressive tf
                print("Non-autoregressive cannot do teacher-forcing")
                return None

        return output, state


class AttnDecoder(nn.Module):
    """ TO DO: Implement your Decoder with Attention """
    def __init__(self, vocab_size, hidden_size, num_layers, max_len, ar=None, tf=None):
        super(AttnDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        """ TO DO: Implement your LSTM """
        self.rnn = YOURLSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(2*hidden_size, vocab_size),
            nn.LogSoftmax(dim=-1))

        self.autoregressive = ar  # True / False
        self.teacher_forcing = tf  # True / False

    def forward(self, max_len, tgt, enc_outputs, state, device): # enc_outputs: (128, 20, 512)
        """ TO DO: feed the input x to Decoder """
        eos = torch.tensor([2]).repeat(state[0].shape[1]).to(device)  # (128,)
        x = self.embedding(eos.clone())  # (128, 512)
        output = []

        for i in range(max_len):
            if i == 0:
                state = self.rnn(x.clone(), state)
            else:
                if self.autoregressive and not self.teacher_forcing:
                    # print("Autoregressive O, Teacher Forcing X, Attention X")
                    prev_out = torch.argmax(output[i - 1], dim=1)
                elif self.autoregressive and self.teacher_forcing:
                    # print("Autoregressive O, Teacher Forcing O, Attention X")
                    prev_out = tgt[:, i - 1]
                elif not self.autoregressive:
                    # print("Autoregressive X, Teacher Forcing X, Attention X")
                    prev_out = torch.argmax(x, dim=1)  # input eos as input every time
                inp = self.embedding(prev_out.clone())
                state = self.rnn(inp.clone(), state)

            s_t = state[0][3] # s_t: (128, 512)
            e_t = torch.bmm(enc_outputs, s_t.unsqueeze(2))  # enc_outputs: (128, 20, 512), state[0][3].unsqueeze(2): (128, 512,1), e_t: (128, 20, 1)
            m = torch.nn.Softmax(dim=1) # softmax function
            alpha_t = m(e_t) # alpht_t: (128, 20, 1)
            tmp_a = torch.bmm(alpha_t.transpose(1,2), enc_outputs) # tmp_a: (128, 1, 512)

            res = torch.concat([tmp_a.squeeze(), s_t], dim=1) # res: (128, 1024)

            out = self.classifier(res.clone()) # out: (128, 24999)
            output.append(out.clone())
        return output, state
