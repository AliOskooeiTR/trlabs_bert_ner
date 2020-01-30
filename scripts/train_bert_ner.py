import pandas as pd
import math
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertConfig
from transformers import BertForTokenClassification
import torch
from torch.optim import Adam
from torch.utils.data import (TensorDataset, DataLoader,
                              RandomSampler, SequentialSampler)
from bert_ner.preprocessing import SentenceGetter
from bert_ner.utils import tag2idx
from bert_ner.train_and_eval import train_model, eval_model
import sys

data_path = "data/"
data_file_address = "data/ner_dataset.csv"
vocabulary = "data/vocab.txt"
output_eval_file_path = os.path.join('data/results', "eval_results.txt")
max_len = 45

df_data = pd.read_csv(
    data_file_address, sep=",",
    encoding="latin1"
).fillna(method='ffill')
df_data.columns
df_data.Tag.unique()

# Get full document data struce
getter = SentenceGetter(df_data)

# Get sentence data
sentences = [[s[0] for s in sent] for sent in getter.sentences]
sentences[0]

# Get pos data
poses = [[s[1] for s in sent] for sent in getter.sentences]
print(poses[0])

# Get tag labels data
labels = [[s[2] for s in sent] for sent in getter.sentences]
print(labels[0])


# Get tag labels data and convert tags to indices
tags_vals = list(set(df_data["Tag"].values))

# Add X  label for word piece support
# Add [CLS] and [SEP] as BERT need
tags_vals.append('X')
tags_vals.append('[CLS]')
tags_vals.append('[SEP]')
tags_vals = set(tags_vals)


# Mapping index to name
tag2name = {tag2idx[key]: key for key in tag2idx.keys()}


# load tokenizer, with manual file address or pretrained address
tokenizer = BertTokenizer(vocab_file=vocabulary, do_lower_case=False)

tokenized_texts = []
word_piece_labels = []
i_inc = 0
for word_list, label in (zip(sentences, labels)):
    temp_lable = []
    temp_token = []

    # Add [CLS] at the front
    temp_lable.append('[CLS]')
    temp_token.append('[CLS]')

    for word, lab in zip(word_list, label):
        token_list = tokenizer.tokenize(word)
        for m, token in enumerate(token_list):
            temp_token.append(token)
            if m == 0:
                temp_lable.append(lab)
            else:
                temp_lable.append('X')

    # Add [SEP] at the end
    temp_lable.append('[SEP]')
    temp_token.append('[SEP]')

    tokenized_texts.append(temp_token)
    word_piece_labels.append(temp_lable)

    if 5 > i_inc:
        print("No.%d,len:%d" % (i_inc, len(temp_token)))
        print("texts:%s" % (" ".join(temp_token)))
        print("No.%d,len:%d" % (i_inc, len(temp_lable)))
        print("lables:%s" % (" ".join(temp_lable)))
    i_inc += 1
# Make text token into id
input_ids = pad_sequences(
    [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
    maxlen=max_len, dtype="long", truncating="post", padding="post"
)

tags = pad_sequences(
    [[tag2idx.get(l) for l in lab] for lab in word_piece_labels],
    maxlen=max_len, value=tag2idx["O"], padding="post",
    dtype="long", truncating="post"
)
print(tags[0])

# For fine tune of predict, with token mask is 1,pad token is 0
attention_masks = [[int(i > 0) for i in ii] for ii in input_ids]
attention_masks[0]

# Since only one sentence, all the segment set to 0
segment_ids = [[0] * len(input_id) for input_id in input_ids]
segment_ids[0]

# split data
tr_inputs, val_inputs, tr_tags, val_tags, tr_masks, val_masks, tr_segs, val_segs = train_test_split(
    input_ids, tags, attention_masks,
    segment_ids, random_state=4, test_size=0.3
)

len(tr_inputs), len(val_inputs), len(tr_segs), len(val_segs)

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)
tr_segs = torch.tensor(tr_segs)
val_segs = torch.tensor(val_segs)
# set batch number
batch_num = 64

# Only set token embedding, attention embedding, no segment embedding
train_data = TensorDataset(
    tr_inputs[0:100],
    tr_masks[0:100],
    tr_tags[0:100]
)
train_sampler = RandomSampler(train_data)
# Drop last can make batch training better for the last one
train_dataloader = DataLoader(
    train_data, sampler=train_sampler,
    batch_size=batch_num, drop_last=True
)

valid_data = TensorDataset(
    val_inputs[0:100],
    val_masks[0:100],
    val_tags[0:100]
)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(
    valid_data, sampler=valid_sampler, batch_size=batch_num)

model_file_address = 'bert-base-cased'

# Will load config and weight with from_pretrained()
model = BertForTokenClassification.from_pretrained(
    model_file_address, num_labels=len(tag2idx)
)


# Set model to GPU,if you are using GPU machine
# model.cuda();


# Add multi GPU support
# if n_gpu >1:
#     model = torch.nn.DataParallel(model)

# Set epoch and grad max num
train_model(
    model,
    train_dataloader,
    tr_inputs,
    epochs=1,
    batch_num=batch_num,
    max_grad_norm=1.0,
    FULL_FINETUNING=True
)


# Evalue loop
eval_model(
    model,
    valid_dataloader,
    val_inputs,
    tag2name,
    output_eval_file_path,
    batch_num=batch_num
)
