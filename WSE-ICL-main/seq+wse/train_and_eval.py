#coding:utf-8
from symbol import file_input
import torch 
from dataset import ZhWikipediaDataSet
import torch
import random
from transformers import BertModel,BertTokenizer
from config import *
import numpy as np
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
from torch.utils.data import DataLoader
from transformers import AdamW
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from seq_model import SeqModel
import torch.nn as nn 
import os 
import time
#from seqeval.metrics import f1_score, precision_score, recall_score
now_time = time.strftime("%Y%m%d%H", time.localtime())


# print(type(config.graident_accumulation_step))
# sys.exit(1)
model = SeqModel(config)
model.to(device)
optimizer = AdamW(model.parameters(),lr = config.learning_rate)

from tqdm import tqdm 
pad =-1.0
def train(model,train_data_loader):
    model.train()
    total_loss,total_accuracy = 0,0 
    for step,batch in enumerate(tqdm(train_data_loader)):
        sent_id,mask,labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)       
#        model.zero_grad()
        logits = model(sent_id,mask)
        loss_fct =  nn.BCEWithLogitsLoss(reduction='none')
        loss_mask = labels != pad
        loss = loss_fct(logits,labels)
        loss_masked = loss.where(loss_mask, torch.tensor(0.0).to(device))
        loss = loss_masked.sum() / loss_mask.sum() 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        #update parameters 
        if (step+1) % config.graident_accumulation_step == 0:
            optimizer.step()
            optimizer.zero_grad()
        #optimizer.step()

        loss_item = loss.item()
        total_loss+=loss_item

    avg_loss = total_loss/ len(train_data_loader)
    return avg_loss 

def evaluate(model,dev_data_loader):
    model.eval()
    from sklearn.metrics import accuracy_score,f1_score, precision_score, recall_score
    total_loss,total_accuracy= 0,0 
    count = 0
    gold_labels = [] 
    preds_list = []
    for step,batch in enumerate(dev_data_loader):
        sent_id,mask,labels = batch[0].to(device),batch[1].to(device),batch[2].to(device)
        logits = model(sent_id,mask)
        loss_fct =  nn.BCEWithLogitsLoss(reduction='none')
        loss_mask = labels != pad
        loss = loss_fct(logits,labels)
        loss_masked = loss.where(loss_mask, torch.tensor(0.0).to(device))
        loss = loss_masked.sum() / loss_mask.sum() 
        #loss = loss_fct(logits,labels)
        loss_item = loss.item()
        sigmoid_fct = torch.nn.Sigmoid()
        preds = (sigmoid_fct(logits)>0.5).int().detach().cpu().numpy()
        gold = batch[2].detach().cpu().numpy()
        for i in range(len(gold)):
            # print(gold[i].tolist())
            # print(gold[i].tolist().index(-1.0))
            index = gold[i].tolist().index(-1.0) if -1.0 in gold[i].tolist() else -1
            if index>0:
                total_accuracy+=accuracy_score(gold[i].tolist()[:index],preds[i].tolist()[:index])/len(gold)
                preds_list.append(preds[i].tolist()[:index])
                gold_labels.append(gold[i].tolist()[:index])
                
            else:
                total_accuracy+=accuracy_score(gold[i].tolist(),preds[i].tolist())/len(gold)
                preds_list.append(preds[i].tolist())
                gold_labels.append(gold[i].tolist())
                
        total_loss+=loss_item/len(gold)

    avg_loss = total_loss/ len(dev_data_loader)
    return avg_loss,total_accuracy/len(dev_data_loader),preds_list,gold_labels




dataset = ZhWikipediaDataSet(filepath=config.train_file,mini_test=False)
train_data_loader = DataLoader(dataset, batch_size=2, collate_fn = collate_fn_wiki, shuffle=True)
dev_dataset = ZhWikipediaDataSet(filepath=config.dev_file)
dev_data_loader =  DataLoader(dev_dataset, batch_size=1, collate_fn = collate_fn_wiki, shuffle=False)
best_acc =0.0 
best_valid_loss = float('inf')

#writer = open('eval_data_PRF.txt','w',encoding='utf-8')
writer = open('eval_data_PRF.txt','w',encoding='utf-8')
for epoch in range(config.epoch):

    print('\n Epoch {:} / {:}'.format(epoch+1 ,config.epoch ))

    train_loss = train(model,train_data_loader)
    dev_loss,dev_acc,preds_list,gold_labels = evaluate(model,dev_data_loader)
    if best_acc<dev_acc:
        best_acc = dev_acc 
        best_valid_loss = dev_loss 
        torch.save(model.state_dict(), "./models/seq_wiki-zh_model_weights_.pth")
        print('save model weith best acc :'+str(best_acc))
    print('train loss {}'.format(train_loss))
    print('val loss {} val acc {}'.format(dev_loss,dev_acc))
    rightpre,entityall,entityres=0,0,0
    for pred,gold in zip(preds_list,gold_labels):
         for l,p in zip(gold,pred):
             if l==1 and p==1:
                 rightpre+=1
             if l==1:
                 entityall+=1
             if p==1:
                 entityres+=1
    if entityres!=0 :
        pP = float(rightpre) / entityres
    else:
        pP =0
    if entityall!=0:
        pR = float(rightpre) /entityall
    else:
        pR =0
    if (pP + pR) !=0:
        pF =(2 * pP * pR) / (pP + pR)
    else:
        pF =0
    print('P {} R {} F {}'.format(pP, pR, pF))
    
    writer.write('train loss '+str(train_loss)+' val loss '+str(dev_loss)+' val acc '+str(dev_acc)+' P '+str(pP)+' R '+ str(pR)+' F '+ str(pF)+'\n')
writer.close() 
