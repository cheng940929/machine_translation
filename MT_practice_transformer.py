import torch
import torch.nn as nn
import nltk
from collections import Counter
import numpy as np

from translate.storage.tmx import tmxfile
big_datapath = "./nmt/en-zh.tmx"


def load_big_data(file_path):
    enu = []
    chs = []
    enu_dev = []
    chs_dev = []

    with open(big_datapath, 'rb') as file:
        tmx_file = tmxfile(file, 'en', 'zh')
        for i,node in enumerate(tmx_file.unit_iter()):
            if i%100 != 0:
                enu.append(["BOS"] + nltk.word_tokenize(node.source.lower()) + ["EOS"])
                chs.append(["BOS"] + [word for word in node.target][:-1] + ["EOS"])
            else:
                enu_dev.append(["BOS"] + nltk.word_tokenize(node.source.lower()) + ["EOS"])
                chs_dev.append(["BOS"] + [word for word in node.target][:-1] + ["EOS"])

        return enu, chs,enu_dev,chs_dev




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

#enu_train,chs_train = load_data(train_file)
#enu_dev,chs_dev = load_data(dev_file)

enu_train,chs_train,enu_dev,chs_dev = load_big_data(big_datapath)




UNK_IDX = 0
PAD_IDX = 1
def build_dic(sources,most_common = 300000):
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
            sentence_with_pad[i] = dic.get(word,0)
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

class PositionalEmbedding(nn.Module):
    def __init__(self,hidden_size,dropout=0.1,maxlen=500):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(maxlen,hidden_size)
        position = torch.arange(maxlen).unsqueeze(1) #max_len*1
        self.dropout = nn.Dropout(dropout)
        W = torch.exp(torch.arange(0,hidden_size,2)*(-torch.log(torch.tensor(10000.))/hidden_size))
        pe[:,0::2] = torch.sin(position*W)
        pe[:,1::2] = torch.cos(position*W)
        pe = pe.unsqueeze(0) # 增加batch维度
        self.register_buffer('pe',pe)

    def forward(self,embed):
        positional_embed = embed+self.pe[:,embed.size()[1],:]
        return self.dropout(positional_embed)



class MultiHeadSelfAttention(nn.Module):
    def __init__(self,hidden_size,dropout = 0.2): #hidden_size = 512
        super(MultiHeadSelfAttention, self).__init__()
        self.wq = nn.Linear(hidden_size,hidden_size)
        self.wk = nn.Linear(hidden_size, hidden_size)
        self.wv = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size
        self.num_head = 8
        self.head_size = int(self.hidden_size/self.num_head)
        self.dropout = nn.Dropout(dropout)


    def divide_to_multi_head(self,x):
        # x.shape = [batch_size,seq_len,all_hidden_size]
        new_size = x.size()[:2]+(self.num_head,self.head_size) # [batch_size,seq_len,num_head,head_size]
        x = x.view(*new_size)
        return x.permute(0,2,1,3)



    def forward(self,q,k,v,mask = None):
        # q,k,v.shape = [batch_size,seq_len,hidden_size(512)]
        Q = self.wq(q) #[batch_size,seq_len,hidden_size(512)]
        K = self.wk(k) #[batch_size,seq_len,hidden_size(512)]
        V = self.wv(v) #[batch_size,seq_len,hidden_size(512)]
        Q_multihead = self.divide_to_multi_head(Q) # [batch_size,num_head,seq_len,head_size]
        K_multihead = self.divide_to_multi_head(K) # [batch_size,num_head,seq_len,head_size]
        V_multihead = self.divide_to_multi_head(V) # [batch_size,num_head,seq_len,head_size]

        QK = torch.matmul(Q_multihead,K_multihead.transpose(-1,-2)) # [batch_size,num_head,seq_len,seq_len]
        if mask is not None:
            QK = QK.masked_fill(mask,-1e6)
        # todo: add padding mask
        QK = QK/((self.head_size)**0.5)
        attention_score = self.dropout(torch.softmax(QK,dim=-1)) #[batch_size,num_head,seq_len,seq_len]
        context = torch.matmul(attention_score,V_multihead).transpose(1,2).contiguous()  #[batch_size,seq_len,num_head,head_size]
        context_new_shape = context.size()[:2]+(self.hidden_size,)
        out = context.view(*context_new_shape)
        return out

class TransformerBlock(nn.Module):
    def __init__(self,hidden_size,expansion = 4,dropout=0.2):
        super(TransformerBlock, self).__init__()
        self.laynorm1 = nn.LayerNorm(hidden_size)
        self.laynorm2 = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadSelfAttention(hidden_size,dropout)
        self.feed_forward = nn.Sequential(nn.Linear(hidden_size,hidden_size*expansion),nn.ReLU(),nn.Linear(hidden_size*expansion,hidden_size))
        self.dropout = nn.Dropout(dropout)
    def forward(self,q,k,v,mask=None):
        attention_out = self.dropout(self.laynorm1(self.self_attention(q,k,v,mask)+q))
        out = self.dropout(self.laynorm2(self.feed_forward(attention_out)+attention_out))
        return out

class Encoder(nn.Module):
    def __init__(self,hidden_size,vocab_size,num_layers):
        super(Encoder, self).__init__()
        self.embed = nn.Embedding(vocab_size,hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(hidden_size) for _ in range(num_layers)])
        self.positional_embed = PositionalEmbedding(hidden_size)


    def forward(self,x):  # x.shape = [batch_size,seq_len]
        block_input = self.embed(x) # [batch_size,seq_len,hidden_size]
        block_input = self.positional_embed(block_input)

        # todo: add positional embedding
        for layer in self.layers:
            block_input = layer(block_input,block_input,block_input,mask = None)
        return block_input


class DecoderBlock(nn.Module):
    def __init__(self,hidden_size):
        super(DecoderBlock, self).__init__()
        self.masked_attention = MultiHeadSelfAttention(hidden_size)
        self.transformer_block = TransformerBlock(hidden_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)




    def forward(self,q,mask,encoder_k,encoder_v):
        masked_attention_out = self.norm1(self.masked_attention(q,q,q,mask)+q)
        out = self.transformer_block(masked_attention_out,encoder_k,encoder_v,mask=None)
        return out

class Decoder(nn.Module):
    def __init__(self,decoder_vocab_size,hidden_size,num_layers):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(decoder_vocab_size,hidden_size)
        self.layers = nn.ModuleList([DecoderBlock(hidden_size) for i in range(num_layers)])
        self.dense = nn.Linear(hidden_size,decoder_vocab_size)
        self.positional_embed = PositionalEmbedding(hidden_size)

    def forward(self,decoder_input,encoder_k,encoder_v,mask):
        decoder_input = self.embed(decoder_input)
        decoder_input = self.positional_embed(decoder_input)
        for layer in self.layers:
            decoder_input = layer(decoder_input,mask,encoder_k,encoder_v)
        out = self.dense(decoder_input)
        return out

def generate_mask_for_decoder(length):
    mask = torch.arange(length)[:,None]<torch.arange(length)[None,:]
    return mask


class Transformer(nn.Module):
    def __init__(self,hidden_size,en_vocab_size,decoder_vocab_size,num_layers):
        super(Transformer, self).__init__()
        self.encoder = Encoder(hidden_size,en_vocab_size,num_layers)
        self.decoder = Decoder(decoder_vocab_size,hidden_size,num_layers)




    def forward(self,encoder_input,decoder_input):
        encoder_out = self.encoder(encoder_input)
        mask = generate_mask_for_decoder(decoder_input.shape[1]).to(device)
        decoder_input = self.decoder(decoder_input,encoder_out,encoder_out,mask)
        out = torch.log_softmax(decoder_input,dim=-1)

        return out
    def translate(self,x,y_input,max_len = 20):
        encoder_out = self.encoder(x)
        prediction = []

        for i in range(max_len):
            out = self.decoder(y_input,encoder_out,encoder_out,None)[:,-1,:]
            pre = torch.argmax(out,dim=-1)
            prediction.append(chs_idx_to_word[int(pre)])
            if chs_idx_to_word[int(pre)] == "EOS":
                return prediction
            y_input = torch.cat([y_input,pre.view(-1,1)],dim = -1)

        return prediction

class LanguageModelLoss(nn.Module):
    def __init__(self):
        super(LanguageModelLoss, self).__init__()


    def forward(self,pre,label,mask):
        # pre.shape = (batch_size,seq_len,vocab_size)
        # label.shape = (batch_size,seq_len,1)
        # mask.shape = (batch_size,seqlen)
        #print(pre.shape,label.shape,mask.shape)
        pre = pre.view(-1,pre.size(2))
        label = label.contiguous().view(-1,1)
        mask = mask.view(-1,1)
        out = -pre.gather(1,label)*mask
        out = torch.sum(out)/torch.sum(mask)
        return out


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Transformer(512,enu_vocab_size,chs_vocab_size,2).to(device)
loss_fn = LanguageModelLoss().to(device)
optimizer = torch.optim.Adam(model.parameters())
model_path = "./mt_transformer_model.pth"

from torch.optim.lr_scheduler import LambdaLR
def get_customized_schedule_with_warmup(optimizer, num_warmup_steps, d_model=1.0, last_epoch=-1):
    def lr_lambda(current_step):
        current_step += 1
        arg1 = current_step ** -0.5
        arg2 = current_step * (num_warmup_steps ** -1.5)
        return (d_model ** -0.5) * min(arg1, arg2)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

scheduler = get_customized_schedule_with_warmup(
    optimizer,
    num_warmup_steps=20000,
    d_model=512
)



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
            pre = model(mb_x,mb_y_input)

            mask = torch.arange(mb_y_output.size()[1],device = device)[None,:]<mb_y_len[:,None]
            loss = loss_fn(pre,mb_y_output,mask)

            optimizer.zero_grad()
            loss.backward()
            scheduler.step()
            optimizer.step()
            print("epoch:{} iter:{} loss:{}".format(epoch,i,loss))

            if i%10 == 0:
                result = eval(model,dev_data)
                print("eval_loss", result)
                if result<eval_loss:
                    eval_loss = result
                    torch.save(model.state_dict(),model_path)

            model.train()


def eval(model,data):
    model.eval()
    loss = 0
    with torch.no_grad():
        for i,(mb_x,mb_x_len,mb_y,mb_y_len) in enumerate(data):
            mb_x = torch.from_numpy(mb_x).to(device)
            mb_x_len = torch.from_numpy(mb_x_len).to(device)
            mb_y= torch.from_numpy(mb_y).to(device)
            mb_y_input = mb_y[:,:-1].to(device)
            mb_y_output = mb_y[:,1:].to(device)
            mb_y_len = torch.from_numpy(mb_y_len-1).to(device)
            pre = model(mb_x,mb_y_input)
            mask = torch.arange(mb_y_output.size()[1],device = device)[None,:]<mb_y_len[:,None]
            loss = loss+loss_fn(pre,mb_y_output,mask)

        return loss/len(data)

train(model,batch_data,30)

model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))


for index in range(100):
    #dev_data = batch_data
    sentence_pool = torch.from_numpy(dev_data[0][0][index]).unsqueeze(0)
    sentence_pool_chs = torch.from_numpy(dev_data[0][2][index]).unsqueeze(0)
    senten_len = torch.from_numpy(np.array([dev_data[0][1][index]]))
    y_input = torch.tensor([[chs_dic["BOS"]]])
    result = model.translate(sentence_pool,y_input)

    a = [enu_idx_to_word[int(i.numpy())] for i in sentence_pool[0]]
    b = [chs_idx_to_word[int(i.numpy())] for i in sentence_pool_chs[0]]
    print(a)
    print(b)
    print(result)








