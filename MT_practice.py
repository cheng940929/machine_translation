import nltk
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F




train_file = "./nmt/en-cn/train.txt"
dev_file = "./nmt/en-cn/dev.txt"

def load_data(file_path):
    enu = []
    chs = []

    with open(file_path) as f:
        for line in f:
            sentence_pair = line.split("\t")
            enu.append(["BOS"]+ nltk.word_tokenize(sentence_pair[0].lower())+["EOS"])
            chs.append(["BOS"] + [word for word in sentence_pair[1]][:-1] + ["EOS"])
    return enu,chs

enu_train,chs_train = load_data(train_file)
enu_dev,chs_dev = load_data(dev_file)




UNK_IDX = 0
PAD_IDX = 1
def build_dic(sources,most_common = 20000):
    word_dic = Counter()
    for source in sources:
        for sentence in source:
            for word in sentence:
                word_dic[word]+=1
    word_dic = word_dic.most_common(most_common)
    dic = {}
    for i,count,in enumerate(word_dic):
        dic[count[0]] = i+2
    dic["UNK"] = UNK_IDX
    dic["PAD"] = PAD_IDX
    dic_len = len(dic)
    return dic,dic_len

enu_dic,enu_vocab_size= build_dic([enu_train,enu_dev])
chs_dic,chs_vocab_size= build_dic([chs_train,chs_dev])
enu_idx_to_word = {key:value for (value,key) in enu_dic.items()}
chs_idx_to_word = {key:value for (value,key) in chs_dic.items()}


def sentence_word_to_index(sentences,dic):
    sentence_onehot = []
    sentences_lengths = [len(sentence) for sentence in sentences]
    eighty_percent_length = int(np.percentile(sentences_lengths,80))
    #eighty_percent_length = max(sentences_lengths)
    for sentence in sentences:
        sentence_with_pad = [0 for i in range(eighty_percent_length)]
        for i,word in enumerate(sentence):
            if i>=eighty_percent_length:
                break
            sentence_with_pad[i] = dic[word]
        sentence_onehot.append(sentence_with_pad)
    sentences_lengths = [min(lenth,eighty_percent_length) for lenth in sentences_lengths]
    return np.array(sentence_onehot),np.array(sentences_lengths)




def get_mini_batch(batch_size,n,shuffle = True):
    batch_idx = np.arange(0,n,batch_size)
    if shuffle:
        np.random.shuffle(batch_idx)
    mini_batches = []
    for idx in batch_idx:
        mini_batches.append([idx,min(idx+batch_size,n)])
    return mini_batches


def prepare_data(enu,chs,enu_dic,chs_dic):
    batch_data = []
    enu_onehot,enu_len = sentence_word_to_index(enu,enu_dic)
    chs_onehot,chs_len = sentence_word_to_index(chs,chs_dic)

    mini_batch = get_mini_batch(batch_size=64,n = len(enu))
    for batch in mini_batch:
        mb_enu = enu_onehot[batch[0]:batch[1],:]
        mb_chs = chs_onehot[batch[0]:batch[1],:]
        enu_l = enu_len[batch[0]:batch[1]]
        chs_l = chs_len[batch[0]:batch[1]]
        batch_data.append((mb_enu,enu_l,mb_chs,chs_l))
    return batch_data

batch_data = prepare_data(enu_train,chs_train,enu_dic,chs_dic)
dev_data = prepare_data(enu_dev,chs_dev,enu_dic,chs_dic)


class Encoder_p(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,dropout = 0.2):
        super(Encoder_p, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,x_len): # x.shape = (batch_size,seq_len,vocab_size)
        embed = self.embedding(x) # embed.shape = (batch_size,seq_len,embedding_size)
        embed = self.dropout(embed)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed,x_len,batch_first=True,enforce_sorted=False)
        original_idx = packed_embed.unsorted_indices
        packed_out, hid= self.rnn(packed_embed)

        pad_packed_embed,_= nn.utils.rnn.pad_packed_sequence(packed_out,batch_first=True)

        out = pad_packed_embed[original_idx]
        hid1 = hid[0][:,original_idx]
        hid2 = hid[1][:,original_idx]
        return out,(hid1,hid2)




class Decoder_p(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,dropout = 0.2):
        super(Decoder_p, self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size,vocab_size)



    def forward(self,y,y_len,hidden):
        embed = self.dropout(self.embed(y))
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed,y_len,batch_first=True,enforce_sorted=False)
        original_idx = packed_embed.unsorted_indices

        packed_rnn,hidden = self.rnn(packed_embed,hidden)

        out,_ = nn.utils.rnn.pad_packed_sequence(packed_rnn,batch_first=True)

        out = out[original_idx]
        hidden = (hidden[0][:,original_idx],hidden[1][:,original_idx])
        out = self.linear(out) # batch_size*seq_len*vocab_size
        out = F.log_softmax(out,2)
        return out,hidden


class seq_2_seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(seq_2_seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,xlen,y,ylen):
        out,hid = self.encoder(x,xlen)
        out,hid = self.decoder(y,ylen,hid)

        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, x_len):  # x.shape = (batch_size,seq_len,vocab_size)
        embed = self.embedding(x)  # embed.shape = (batch_size,seq_len,embedding_size)
        embed = self.dropout(embed)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, x_len, batch_first=True, enforce_sorted=False)
        #pad_packed_embed, _ = nn.utils.rnn.pad_packed_sequence(packed_embed, batch_first=True) #test
        original_idx = packed_embed.unsorted_indices
        packed_out, hid = self.rnn(packed_embed)

        pad_packed_embed, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        out = pad_packed_embed
        hidden = hid[0][:, original_idx]
        cell = hid[1][:, original_idx]

        hidden = self.dense(torch.cat([hidden[-1], hidden[-2]], dim=-1)).unsqueeze(0)
        cell = self.dense(torch.cat([cell[-1], cell[-2]], dim=-1)).unsqueeze(0)
        # out.shape = (batch_size,seq_len,2*hidden_size)
        # hidden.shape = (1,batch_size,hidden_size)
        return out, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W_a = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, decoder_output, context, mask):
        # context.shape = (batch_size,en_seq_len,2*hidden_size)
        # decoder_output = (batch_size,de_seq_len,hidden_size)
        context_in = self.W_a(context).transpose(1, 2)  # context_in.shape = (batch_size,hidden_size,en_seq_len)

        attention = torch.bmm(decoder_output, context_in)  # attention.shape = (batch_size,de_seq_len,en_seq_len)

        attention = attention.masked_fill(mask, -1e6)

        attention_softmaxed = torch.softmax(attention, dim=2)  # attention.shape = (batch_size,de_seq_len,en_seq_len)

        weighted_sum = torch.bmm(attention_softmaxed,
                                 context)  # weighted_sum.shape = (batch_size,de_seq_len,2*hidden_size)

        return weighted_sum




class Decoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,dropout = 0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.attention = Attention(hidden_size)
        self.dense = nn.Linear(hidden_size*3,vocab_size)




    def forward(self,y,y_len,hidden,context,x_len):
        embed = self.dropout(self.embed(y))
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed,y_len,batch_first=True,enforce_sorted=False)
        original_idx = packed_embed.unsorted_indices

        packed_rnn,hidden = self.rnn(packed_embed,hidden)

        out,_ = nn.utils.rnn.pad_packed_sequence(packed_rnn,batch_first=True)

        #out = out[original_idx]
        mask = self.generate_mask_for_attention(y_len,x_len)
        c_t = self.attention(out,context,mask)
        out = self.dense(torch.tanh(torch.cat([c_t,out],dim=2)))


        hidden = (hidden[0][:,original_idx],hidden[1][:,original_idx])
        out = F.log_softmax(out,2)
        return out,hidden

    def generate_mask_for_attention(self,x_len,y_len):
        # (batch_size, de_seq_len, en_seq_len)
        device = x_len.device
        x_len_max = max(x_len)
        y_len_max = max(y_len)
        x_mask = torch.arange(x_len_max,device=x_len.device)[None,:]<x_len[:,None] # batch_size,max_x
        y_mask = torch.arange(y_len_max,device=x_len.device)[None,:] < y_len[:,None] # batch_size,max_y
        mask = (x_mask[:,:,None]*y_mask[:,None,:]).logical_not().bool()
        return mask


class seq_2_seq_withattention(nn.Module):
    def __init__(self,encoder,decoder):
        super(seq_2_seq_withattention, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,x,xlen,y,ylen):
        en_out,hid = self.encoder(x,xlen)
        out,hid = self.decoder(y,ylen,hid,en_out,xlen)

        return out
    def translate(self,x,y_input,x_len,max_len = 20):
        encoder_out,hid = self.encoder(x,x_len)
        prediction = []
        batch_size = x.shape[0]
        y_len = torch.ones(batch_size)
        y_len = y_len.long()
        for i in range(max_len):
            out,hid = self.decoder(y_input,y_len,hid,encoder_out,x_len)
            y_input = torch.argmax(out,dim=-1).view(batch_size,1)
            prediction.append(chs_idx_to_word[int(y_input[0][0])])
        return prediction

class LanguageModelLoss(nn.Module):
    def __init__(self):
        super(LanguageModelLoss, self).__init__()


    def forward(self,pre,label,mask):
        # pre.shape = (batch_size,seq_len,vocab_size)
        # label.shape = (batch_size,seq_len,1)
        # mask.shape = (batch_size,seqlen)
        pre = pre.view(-1,pre.size(2))
        label = label.contiguous().view(-1,1)
        mask = mask.view(-1,1)

        out = -pre.gather(1,label)*mask
        out = torch.sum(out)/torch.sum(mask)
        return out



embedding_size = 500
hidden_size = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(enu_vocab_size,embedding_size,hidden_size)
decoder = Decoder(chs_vocab_size,embedding_size,hidden_size)
model = seq_2_seq_withattention(encoder,decoder).to(device)

loss_fn = LanguageModelLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())
model_path = "./mt_model.pth"
def train(model,data,epochs):
    model.train()
    eval_loss = 100
    for epoch in range(epochs):
        for i,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device)
            mb_x_len = torch.from_numpy(mb_x_len).to(device)
            mb_y= torch.from_numpy(mb_y).to(device)
            mb_y_input = mb_y[:,:-1].to(device)
            mb_y_output = mb_y[:,1:].to(device)
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device)
            pre = model(mb_x,mb_x_len,mb_y_input,mb_y_len)
            mask = torch.arange(mb_y_len.max(),device = device)[None,:]<mb_y_len[:,None]
            loss = loss_fn(pre,mb_y_output,mask)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(),5.)
            optimizer.step()
            print("epoch:{} iter:{} loss:{}".format(epoch,i,loss))

            if i%10 == 0:
                result = eval(model,dev_data)
                if result<eval_loss:
                    eval_loss = result
                    torch.save(model.state_dict(),model_path)

            model.train()


def eval(model,data):
    model.eval()
    with torch.no_grad():
        for i,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device)
            mb_x_len = torch.from_numpy(mb_x_len).to(device)
            mb_y= torch.from_numpy(mb_y).to(device)
            mb_y_input = mb_y[:,:-1].to(device)
            mb_y_output = mb_y[:,1:].to(device)
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device)
            pre = model(mb_x,mb_x_len,mb_y_input,mb_y_len)
            mask = torch.arange(mb_y_len.max(),device = device)[None,:]<mb_y_len[:,None]
            loss = loss_fn(pre,mb_y_output,mask)
            print("eval_loss",loss)

            return loss

train(model,batch_data,20)

model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))


for index in range(15):
    #dev_data = batch_data
    sentence_pool = torch.from_numpy(dev_data[0][0][index]).unsqueeze(0)
    sentence_pool_chs = torch.from_numpy(dev_data[0][2][index]).unsqueeze(0)
    senten_len = torch.from_numpy(np.array([dev_data[0][1][index]]))
    y_input = torch.tensor([[chs_dic["BOS"]]])
    result = model.translate(sentence_pool,y_input,senten_len)

    a = [enu_idx_to_word[int(i.numpy())] for i in sentence_pool[0]]
    b = [chs_idx_to_word[int(i.numpy())] for i in sentence_pool_chs[0]]
    print(a)
    print(b)
    print(result)



















