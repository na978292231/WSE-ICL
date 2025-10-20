import json
import random
from datetime import datetime
from time import sleep
import logging
import argparse
from tqdm import tqdm
import csv
import os
import psutil
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import GPT2Config

from utils.dataset import *
from utils.template import *
from utils.trainset import TrainsetStore
from utils.anchor import AnchorStore
#from module_interaction import MyMultiHeadAttention
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import time
from seq_model import SeqModel
from config import *
from transformers import BertTokenizer
def get_memory_usage():
    # Get memory usage on the CPU
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / (1024 ** 2)  # Convert to MB
    return memory_usage

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        # Get memory usage on the GPU (CUDA device)
        return torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    else:
        return 0
seq_tokenizer = BertTokenizer.from_pretrained('/data/llmweights/bert/bert-cn/chinese-bert-wwm-ext')
seq_model = SeqModel(config)

seq_model.load_state_dict(torch.load('./models/seq_wiki-zh_model_weights_.pth'),False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_model.to(device)
names = [
    "Nearest Neighbors",
    "Linear SVM",
    "Decision Tree",
    "MLP",
    "Naive Bayes",
    "Logistic Regression"
]

classifiers = [
    KNeighborsClassifier(n_neighbors=3),
    LinearSVC(random_state=0, multi_class='crammer_singer'),
    DecisionTreeClassifier(random_state=0),
    MLPClassifier(random_state=0,max_iter=1000),
    GaussianNB(),
    LogisticRegression(random_state=0, multi_class='multinomial'),
]


os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To avoid warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="KNN Prompting.")
    parser.add_argument(
        "--llm_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_train_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_demo_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_anchor_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--knn",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--feature_choose",
        type=str,
        choices=[' only_label', 'fuzzy_label', 'all'],
        default=' only_label',
    )
    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    return args


def llm_gen(model, prompt, tokenizer, max_context_len,sentence, csv_writer):
    cpu_memory_before = get_memory_usage()
    gpu_memory_before = get_gpu_memory_usage()
    start_time = time.time()
    inputs = tokenizer.encode_plus(prompt, return_tensors="pt", padding=True).to(device=model.device)
    if inputs['input_ids'].shape[1] > max_context_len:
        inputs['input_ids'] = inputs['input_ids'][:, -max_context_len:]
        inputs['attention_mask'] = inputs['attention_mask'][:, -max_context_len:]
    seq_inputs = seq_tokenizer.encode_plus(sentence, return_tensors="pt", padding=True).to(device=model.device)
    if seq_inputs['input_ids'].shape[1] > config.max_seqence_length:
        seq_inputs['input_ids'] = seq_inputs['input_ids'][:, -config.max_seqence_length:]
        seq_inputs['attention_mask'] = seq_inputs['attention_mask'][:, -config.max_seqence_length:]
    logits = seq_model(seq_inputs['input_ids'],seq_inputs['attention_mask']).detach().cpu()
    embedding_dim = 256
    embedding_dim2 = model.config.hidden_size
    #fc = nn.Linear(embedding_dim, embedding_dim2).cpu()
    sh=logits
    sh=torch.mean(sh,dim=1)
   
    with torch.no_grad():
        hs = model.forward(input_ids=inputs['input_ids'],
                               attention_mask=inputs['attention_mask'],output_hidden_states =True,
                               return_dict=True).hidden_states[-1].detach().cpu()
    # the output prob is shifted by -1, so we should use the output at the last input token position
    # gen_hs.shape = [1, 1600]
    hs= hs[:, -1, :]
    att=torch.matmul(sh.transpose(1,0),hs)
    gen_hs_p =torch.matmul(sh,att)
    cpu_memory_after = get_memory_usage()
    gpu_memory_after = get_gpu_memory_usage()
    end_time= time.time()
    generation_time = end_time - start_time
    cpu_memory_usage = cpu_memory_after - cpu_memory_before
    gpu_memory_usage = gpu_memory_after - gpu_memory_before
    print(f"Generation Time: {generation_time:.4f}s")
    print(f"CPU Memory Usage: {cpu_memory_usage:.2f} MB")
    print(f"GPU Memory Usage: {gpu_memory_usage:.2f} MB")
    csv_writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), generation_time, cpu_memory_usage, gpu_memory_usage])

    return gen_hs_p


def get_fuzzy_label_ids(model, tokenizer, id2verb, k=10):
    from transformers import GPT2PreTrainedModel
    embeddings = model.transformer.wte.weight.detach() if isinstance(model, GPT2PreTrainedModel) \
                    else model.model.embed_tokens.weight.detach()

    def l2norm(X, dim=-1, eps=1e-8):
        """L2-normalize columns of X"""
        norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
        X = torch.div(X, norm)
        return X
    
    embeddings = l2norm(embeddings)
    queryT = torch.transpose(embeddings, 0, 1)
    similarity = torch.mm(embeddings, queryT)
    # print(similarity.shape)

    vocab = [v for v,i in tokenizer.get_vocab().items()]

    fuzzy_label_ids = []
    for label_verb in id2verb:
        label_verb_token_id = tokenizer.encode(' ' + label_verb)[-1] # note the space before label word
        similar_value, similar_id = torch.topk(similarity[label_verb_token_id], k=k)
        # print(label_verb, vocab[label_verb_token_id], list(zip(similar_value.tolist(), [vocab[i] for i in similar_id.tolist()])))

        fuzzy_label_ids.extend(similar_id.tolist())
    
    return list(set(fuzzy_label_ids))

def main():
    args = parse_args()

    args.n_anchor_shot = args.n_train_shot - args.n_demo_shot
    if args.n_anchor_shot <= 0:
        raise Exception("Num. of demonstration must be set smaller than num. of training.")

    setup_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.llm_dir, use_fast=False)
    # set pad token ids for batched inference cus gpt2 does not have one
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model_config = AutoConfig.from_pretrained(args.llm_dir)
    if "30" in args.llm_dir :
        model = AutoModelForCausalLM.from_pretrained(args.llm_dir, device_map='auto') # , device_map='auto'
    elif "70" in args.llm_dir:
        model = AutoModelForCausalLM.from_pretrained(args.llm_dir, device_map='auto',load_in_8bit=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.llm_dir)
        model.to(device)
    # model.to(device)
    model.eval()

    if 'gpt2' in args.llm_dir:
        max_context_len = 1024
    else:
        max_context_len = 4096 if 'llama2' in args.llm_dir else 2048

    # prepare dataset
    if args.dataset == 'sst2':
        AutoDataset = SST2Dataset
    elif args.dataset == 'subj':
        AutoDataset = SUBJDataset
    elif args.dataset == 'agnews':
        AutoDataset = AGNEWSDataset
    elif args.dataset == 'cb':
        AutoDataset = CBDataset
    elif args.dataset == 'cr':
        AutoDataset = CRDataset
    elif args.dataset == 'dbpedia':
        AutoDataset = DBPEDIADataset
    elif args.dataset == 'mpqa':
        AutoDataset = MPQADataset
    elif args.dataset == 'mr':
        AutoDataset = MRDataset
    elif args.dataset == 'rte':
        AutoDataset = RTEDataset
    elif args.dataset == 'trec':
        AutoDataset = TRECDataset
    elif args.dataset == 'clause':
        AutoDataset = CLAUSEDataset
    elif args.dataset == 'wiki-zh':
        AutoDataset = WIKIZHDataset
    logger.info(f"===== {args.dataset} / {args.n_train_shot} / {args.seed} =====")

    datadir = os.path.join(args.data_dir, args.dataset)
    train_data = AutoDataset(datadir, mode='train')
    dev_data = AutoDataset(datadir, mode='dev')

    anchor_data = AutoDataset(datadir, mode='train')

    # Stage1: Meta Test
    train_data.subsamplebyshot(args.n_demo_shot, args.seed)
    prompt_prefix = make_prompt(train_data, args.dataset, mode='train')
    anchor_data.subsamplebyshot(args.n_anchor_shot, args.seed, exclude=train_data.data)
    label2id = dev_data.label2id
    id2verb = train_data.id2verb
    
    # Stage1: train set

    logger.info(f"===== build train set store of {train_data.__len__()} examples =====")
    trainset_store = TrainsetStore(K=anchor_data.__len__(),
                               dim= model_config.n_embd if isinstance(model_config, GPT2Config) else model_config.hidden_size,
                               n_class=len(label2id))
    with open('memory_usage_train_log_{}_train_shot_{}_demo_shot_{}_seed.csv'.format(args.n_train_shot,args.n_demo_shot,args.seed), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Timestamp', 'Generation Time (s)', 'CPU Memory Usage (MB)', 'GPU Memory Usage (MB)'])
        
        for ins in tqdm(anchor_data.data, total=anchor_data.__len__()):
            labels = label2id[ins['label']]
            prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
            gen_hs = llm_gen(model, prompt, tokenizer, max_context_len,ins['sentence'],csv_writer)
            trainset_store.enqueue(gen_hs, torch.tensor(labels))

    # Stage2: dev set
    logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
    dev_labels = []
    dev_pred = []
    with open('memory_usage_dev_log_{}_train_shot_{}_demo_shot_{}_seed.csv'.format(args.n_train_shot,args.n_demo_shot,args.seed), 'w', newline='') as csvfile2:
        csv_writer2 = csv.writer(csvfile2)
        csv_writer2.writerow(['Timestamp', 'Generation Time (s)', 'CPU Memory Usage (MB)', 'GPU Memory Usage (MB)'])
        for ins in tqdm(dev_data.data[:256], total=256):
            dev_labels.append(label2id[ins['label']])
            prompt = prompt_prefix + make_prompt(ins, args.dataset, mode='inference')
            gen_hs = llm_gen(model, prompt, tokenizer, max_context_len,ins['sentence'],csv_writer2)
            dev_pred.append(gen_hs)
           


    # Stage 3: dev set classification
    
    def classification_by_feature(feature_ids, feature_name=args.feature_choose):
        chosen_train_x, chosen_train_y = trainset_store.get_data(feature_ids)
        test_x = [all_feature[0, feature_ids].cpu().numpy().tolist() for all_feature in dev_pred]
        test_y = dev_labels

        logger.info(f"===== classification by {feature_name} =====")
        for name, clf in zip(names, classifiers):
            candidate_methods_for_all_feature = ["Nearest Neighbors", "MLP", "Logistic Regression"]
            if feature_name == 'all' and  name not in candidate_methods_for_all_feature:
                continue
            logger.info(f"Classification by {name}")
            start_time =  time.time()
            clf = make_pipeline(StandardScaler(), clf)
            clf.fit(chosen_train_x, chosen_train_y)
            score = clf.score(test_x, test_y)
            end_time =  time.time()
            running_time = end_time - start_time
            logger.info(f"accuracy_score: {score}; running_time: {running_time}s")

            # logging
            save_results_file = os.path.join(args.output_dir, 'results_hs_as_feature.csv'.format(feature_name))
            csv_exists = os.path.isfile(save_results_file)
            with open(save_results_file, 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                if not csv_exists:
                    csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'n_demo_shot', 'method', 'seed', 'acc', 'running_time'])
                csvwriter.writerow([args.dataset,
                                    args.llm_dir,
                                    args.n_train_shot,
                                    args.n_demo_shot,
                                    name,
                                    args.seed,
                                    score,
                                    running_time])
    
    # chosen feature's index for classification
    chosen_feature_indexes = list(range(model_config.n_embd if isinstance(model_config, GPT2Config) else model_config.hidden_size))
    classification_by_feature(chosen_feature_indexes)
    



if __name__ == "__main__":
    main()
