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



class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout=0.2):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True,num_layers=2)
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, x_len):  # x.shape = (batch_size,seq_len,vocab_size)
        embed = self.embedding(x)  # embed.shape = (batch_size,seq_len,embedding_size)
        embed = self.dropout(embed)
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, x_len, batch_first=True, enforce_sorted=False)
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
        self.W_b = nn.Linear(hidden_size,hidden_size,bias=False)
        self.W_c = nn.Linear(hidden_size,1)
        self.dense = nn.Linear(2*hidden_size,hidden_size)

    def forward(self, decoder_hidden, context, mask):
        # context.shape = (batch_size,encoder_seq_len,2*hidden_size)
        # decoder_hidden.shape = (1,batch_size,hidden_size)
        context_in = self.W_a(context)  # context_in.shape = (batch_size,en_seq_len,hidden_size)
        hidden_in = self.W_b(decoder_hidden).transpose(0,1) # hidden_in.shape = (batch_size,1,hidden_size)
        attention = self.W_c(torch.tanh(context_in+hidden_in))
        # attention.shape = (batch_size,en_seq_len,1)
        attention = attention.masked_fill(mask,-1e6)
        attention_softmaxed = torch.softmax(attention,1)
        weighted_sum = torch.bmm(attention_softmaxed.transpose(1,2),context)
        # weighted_sum.shape = (batch_size,1,2*hidden_size)
        '''
        weighted_sum = context*attention_softmaxed # weighted_sum.shape = (batch_size,encoder_seq_len,2*hidden_size)

        context_out = self.dense(torch.sum(weighted_sum,dim=1)) # context_out = (batch_size,1,2*hidden_size)
        return context_out
        '''

        return self.dense(weighted_sum)





class Decoder(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size,dropout = 0.2):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(vocab_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size+hidden_size,hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size,vocab_size)
        self.attention = Attention(hidden_size)
        self.dense = nn.Linear(hidden_size,vocab_size)




    def forward(self,y,hidden,context,x_len):
        # Decdoer会一个单词一个单词的解码，所以这里的y就是一个单词，而不是像以前把一句话都传进来。
        # y.shape = (batch_size,1)
        embed = self.dropout(self.embed(y)) # embed.shape = (batch_size,1,embeding_size)
        mask = torch.arange(max(x_len), device=x_len.device)[None, :] < x_len[:, None]
        mask = mask.logical_not()
        c_t = self.attention(hidden[0], context, mask.unsqueeze(2)).squeeze(1) #c_t.shape = (batch_size,1,hidden_size)
        #print(c_t.shape,embed.shape)
        lstm_in = torch.cat([c_t,embed],dim=-1).unsqueeze(1) # lstm_in.shape = (batch_size,1,hidden_size+embeding_size)
        out,hid = self.rnn(lstm_in,hidden)
        # out.shape = (batch_size,1,hidden_size)
        # hid.shape = (1,batch_size,hidden_size)
        out = self.dense(out) # out.shape = (batch_size,1,vocab_size)
        out = torch.nn.functional.log_softmax(out,dim = -1)
        return out,hid

class LanguageModelLoss(nn.Module):
    def __init__(self):
        super(LanguageModelLoss, self).__init__()


    def forward(self,pre,label,mask):
        pre = pre.squeeze()
        label = label.unsqueeze(1)
        # pre.shape = (batch_size,vocab_size)
        # label.shape = (batch_size,1)
        # mask.shape = (batch_size,1)

        out = -pre.gather(1,label)*mask
        out = torch.sum(out)/torch.sum(mask)
        return out



embedding_size = 1000
hidden_size = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder(enu_vocab_size,embedding_size,hidden_size).to(device)
decoder = Decoder(chs_vocab_size,embedding_size,hidden_size).to(device)


loss_fn = LanguageModelLoss().to(device)
optimizer = torch.optim.Adam([{'params':encoder.parameters()},{'params':decoder.parameters()}])
#optimizer = torch.optim.Adam(encoder.parameters())
encoder_path = "./mt_encoder_model.pth"
decoder_path = "./mt_decoder_model.pth"
def train_step(mb_x,x_len,mb_y):
    y_len = mb_y.shape[1]
    encoder_out,encoder_hid = encoder(mb_x,x_len)
    decoder_hid = encoder_hid
    decoder_input = mb_y[:,0]
    batch_loss = 0
    '''
        for i in range(1,y_len):
        if i !=1:
            encoder_out, encoder_hid = encoder(mb_x, x_len)

        decoder_out, _ = decoder(decoder_input,decoder_hid,encoder_out,x_len)
        #decoder_hid = _
        decoder_hid = (_[0].detach(),_[1].detach())
        # mask.shape = (batch_size,1)
        mask = torch.eq(decoder_input,torch.tensor(0)).logical_not().bool().unsqueeze(1)
        decoder_input = mb_y[:,i]

        loss = loss_fn(decoder_out,decoder_input,mask)
        batch_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        print('yes')
        #nn.utils.clip_grad_norm()
        optimizer.step()
    '''
    for i in range(1,y_len):
        '''
        if i !=1:
            encoder_out, encoder_hid = encoder(mb_x, x_len)
            '''

        decoder_out, _ = decoder(decoder_input,decoder_hid,encoder_out,x_len)
        #decoder_hid = _
        decoder_hid = (_[0].detach(),_[1].detach())
        # mask.shape = (batch_size,1)
        mask = torch.eq(decoder_input,torch.tensor(0)).logical_not().bool().unsqueeze(1)
        decoder_input = mb_y[:,i]

        loss = loss_fn(decoder_out,decoder_input,mask)
        batch_loss+=loss
    optimizer.zero_grad()
    batch_loss.backward()

    optimizer.step()
    return batch_loss

def eval_step(mb_x,x_len,mb_y):
    y_len = mb_y.shape[1]
    encoder_out,encoder_hid = encoder(mb_x,x_len)
    decoder_hid = encoder_hid
    decoder_input = mb_y[:,0]
    batch_loss = 0
    for i in range(1,y_len):
        if i !=1:
            encoder_out, encoder_hid = encoder(mb_x, x_len)
        decoder_out, decoder_hid = decoder(decoder_input,decoder_hid,encoder_out,x_len)
        #decoder_hid = (_[0].detach(),_[1].detach())
        # mask.shape = (batch_size,1)
        mask = torch.eq(decoder_input,torch.tensor(0)).logical_not().bool().unsqueeze(1)
        decoder_input = mb_y[:,i]

        loss = loss_fn(decoder_out,decoder_input,mask)
        batch_loss+=loss
    return batch_loss



def train(encoder,decoder,data,epochs):
    encoder.train()
    decoder.train()
    eval_loss = 2500
    for epoch in range(epochs):
        for i,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device)
            mb_x_len = torch.from_numpy(mb_x_len).to(device)
            mb_y= torch.from_numpy(mb_y).to(device)

            batchloss = train_step(mb_x,mb_x_len,mb_y)
            print("epoch:{} iter:{} loss:{}".format(epoch,i,batchloss))
            if i%10 == 0:
                result = eval(encoder,decoder,dev_data)
                if result<eval_loss:
                    eval_loss = result
                    torch.save(encoder.state_dict(),encoder_path)
                    torch.save(decoder.state_dict(), decoder_path)
            encoder.train()
            decoder.train()


def eval(encoder,decoder,data):
    encoder.eval()
    decoder.eval()
    eval_loss = 0
    with torch.no_grad():
        for i,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device)
            mb_x_len = torch.from_numpy(mb_x_len).to(device)
            mb_y = torch.from_numpy(mb_y).to(device)

            eval_loss += eval_step(mb_x, mb_x_len, mb_y)
    print("Eval_loss:", eval_loss)
    return eval_loss


#train(encoder,decoder,batch_data,100)


## translate

encoder.load_state_dict(torch.load(encoder_path,map_location=device))
decoder.load_state_dict(torch.load(decoder_path,map_location=device))
encoder.eval()
decoder.eval()


def translate(x, y_input, x_len, max_len=20):
    encoder_out, hid = encoder(x, x_len)
    prediction = []
    batch_size = x.shape[0]
    y_len = torch.ones(batch_size)
    y_len = y_len.long()
    for i in range(max_len):
        out, hid = decoder(y_input,hid,encoder_out, x_len)
        y_input = torch.argmax(out, dim=-1).view(batch_size)
        prediction.append(chs_idx_to_word[int(y_input[0])])
    return prediction

    
for index in range(15):
    #dev_data = batch_data
    sentence_pool = torch.from_numpy(dev_data[0][0][index]).unsqueeze(0)
    sentence_pool_chs = torch.from_numpy(dev_data[0][2][index]).unsqueeze(0)
    senten_len = torch.from_numpy(np.array([dev_data[0][1][index]]))
    y_input = torch.tensor([chs_dic["BOS"]])
    result = translate(sentence_pool,y_input,senten_len)

    a = [enu_idx_to_word[int(i.numpy())] for i in sentence_pool[0]]
    b = [chs_idx_to_word[int(i.numpy())] for i in sentence_pool_chs[0]]
    print(a)
    #print(b)
    print(result)

'''
    def translate(sentence,a,b,y_len,sentence_len):
    out,hid = encoder(sentence,sentence_len)
    decoder_in = hid
    y_input = torch.from_numpy(np.array([a["BOS"]]))
    translate_out = ''
    for i in range(y_len):
        y_out,decoder_in = decoder(y_input,decoder_in,out,sentence_len)
        pre = torch.argmax(y_out,dim=-1)
        if pre.numpy() == b["EOS"]:
            return translate_out
        translate_out+=chs_idx_to_word[pre.numpy()[0][0]]
        y_input = torch.argmax(y_out,dim=-1).squeeze(0)

    return translate_out
dev_data = batch_data
for index in range(15):
    dev_data = batch_data
    sentence_pool = torch.from_numpy(dev_data[0][0][index]).unsqueeze(0)
    sentence_pool_chs = torch.from_numpy(dev_data[0][2][index]).unsqueeze(0)
    senten_len = torch.from_numpy(np.array([dev_data[0][1][index]]))
    result = translate(sentence_pool,enu_dic,chs_dic,100,senten_len)
    a = [enu_idx_to_word[int(i.numpy())] for i in sentence_pool[0]]
    b = [chs_idx_to_word[int(i.numpy())] for i in sentence_pool_chs[0]]
    print(a)
    #print(b)
    print(result)
    '''
    


























