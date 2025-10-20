#coding:utf-8
import torch.nn as nn 
from transformers import BertModel
from transformers import BertTokenizer,AutoTokenizer


tokenizer = BertTokenizer.from_pretrained('/data/llmweights/bert-kor-base')
import torch 
from tools import Trie
from fastNLP import Vocabulary
from fastNLP.embeddings import BertEmbedding


def load_yangjie_rich_pretrain_word_list(embedding_path, drop_characters=True):
    f = open(embedding_path, 'r', encoding='utf-8')
    lines = f.readlines()
    w_list = []
    for line in lines:
        splited = line.strip().split(' ')
        w = splited[0]
        w_list.append(w)

    if drop_characters:
        w_list = list(filter(lambda x: len(x) != 1, w_list))

    return w_list
word_path = '/home/xzg/cc.ko.300.vec/cc.ko.300_utf8.vec'
#word_char_mix_embedding_path = yangjie_rich_pretrain_char_and_word_path
w_list = load_yangjie_rich_pretrain_word_list(word_path)
w_trie = Trie()
for w in w_list:
    w_trie.insert(w)
vocab = Vocabulary()
vocab.add_word_lst(w_list)
bert_embedding = BertEmbedding(vocab,model_dir_or_name='/data/llmweights/bert-kor-base2',requires_grad=False,auto_truncate=True,
                                           word_dropout=0.01)
class SeqModel(nn.Module):
    def __init__(self,config) -> None:
        super().__init__()
        self.config = config 
        self.lstm_hidden_size = 128 
        self.dropout = nn.Dropout(0.1)
        self.encoder  = BertModel.from_pretrained(pretrained_model_name_or_path=config.pretrain_model_path)
        self.linear = nn.Linear(config.hidden_size,1)
        if config.use_bilstm:
            
            self.lstm = nn.LSTM(input_size=config.hidden_size,hidden_size=self.lstm_hidden_size,num_layers=1,batch_first=True,bidirectional=True)
            # self.lstm = nn.GRU(input_size=config.hidden_size,hidden_size=config.hidden_size,num_layers=1,batch_first=True,bidirectional=True)
            self.linear = nn.Linear(self.lstm_hidden_size*2,1)
        self.W = nn.ModuleList()
        for i in range(2):
            self.W.append(nn.Linear(config.hidden_size, config.hidden_size)).cuda()
        self.bert=bert_embedding
    def forward(self,input_ids,input_mask):
        slist=[]
        elist=[]
        sents_id=[]
        sents_mask=[]
        batch_data = []
        words=[]
        for input_id in input_ids:
            tokened_text = tokenizer.convert_ids_to_tokens(input_id)
            new_text=[]
            for i in tokened_text:
                if i not in ['[CLS]','[SEP]','[PAD]']:
                    new_text.append(i)
            sen="".join(new_text)
            lexicon_in_s, bme_num = w_trie.get_lexicon(sen)
            pos_s = [-1]+list(range(len(sen)))+list(map(lambda x: x[0], lexicon_in_s))+[-2]
            pos_e = [-1]+list(range(len(sen)))+list(map(lambda x: x[1], lexicon_in_s))+[-2]
            word=list(map(lambda x: vocab.to_index(x[2]), lexicon_in_s))
            #word=vocab.to_index(word)
            word=torch.tensor(word)
            while len(word)<self.config.max_seqence_length-len(input_id):
                word=torch.cat((word, torch.Tensor([0])), 0)
            word=word[:self.config.max_seqence_length-len(input_id)]    
            words.append(word)
            
           
            
            pos_s=torch.tensor(pos_s)
            pos_e=torch.tensor(pos_e)
            
            while len(pos_s)<self.config.max_seqence_length:
                
                pos_s=torch.cat((pos_s, torch.Tensor([-3])), 0)
                pos_e=torch.cat((pos_e, torch.Tensor([-3])), 0)
            
            pos_s=pos_s[:self.config.max_seqence_length]
            pos_e=pos_e[:self.config.max_seqence_length]
            
            
            
            
            
            slist.append(pos_s)
            elist.append(pos_e)
        
        ps=torch.stack(slist,0)
        pe=torch.stack(elist,0)
        wordss=torch.stack(words,0)
        last_hidden_c,_= self.encoder(input_ids=input_ids, attention_mask=input_mask)[:2]
        word_embedding=self.bert(wordss.long().cuda())
        last_hidden_cw=torch.cat([last_hidden_c,word_embedding],1)
        hiddens=[]
        num=[]
        for hidden in last_hidden_cw:
            num.append(len(hidden))
            while len(hidden)<self.config.max_seqence_length:
                hidden=torch.cat((hidden, torch.Tensor([0.0]*self.config.hidden_size).unsqueeze(-2).cuda()), 0)
            hiddens.append(hidden)
        last_hidden=torch.stack(hiddens,0)
        pss=ps.unsqueeze(-1)-ps.unsqueeze(-2)
        pee=pe.unsqueeze(-1)-pe.unsqueeze(-2)
        zero = torch.zeros_like(pss)
        one = torch.ones_like(pss)
        ret=pss*pee
        
        ret= torch.where(ret> 0, zero, one)
        
        
        ret=ret.float().cuda()
        #ret=ret.squeeze(0)
        denom=ret.sum(2).unsqueeze(2) + 1
        for i in range(2):
            Ax= torch.matmul(ret,last_hidden)
            AxW = self.W[i](Ax)   ## N x m
            AxW = AxW+self.W[i](last_hidden)  ## self loop  N x h
            AxW = AxW / denom
            last_hidden = torch.relu(AxW)
        #num=torch.count_nonzero(last_hidden, dim=1)
        pooled_output  = torch.mean(last_hidden,dim=1)
        
        #last_hidden_c,_= self.encoder(input_ids=input_ids, attention_mask=input_mask)[:2]
        #pooled_output  = torch.mean(last_hidden,dim=1)
        output = self.dropout(pooled_output)
        
        output = output.view(-1,self.config.max_sentences_num,self.config.hidden_size)
        if self.config.use_bilstm:   
            output, (h_n, c_n) = self.lstm(output)
           
        logits = self.linear(output)
        batch_size =logits.size()[0]
        logits = torch.reshape(logits,(-1,self.config.max_sentences_num))    
        return logits

