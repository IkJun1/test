with open("hobby_data.txt",'r', encoding='utf-8') as f :
    hobby_data = f.readlines()
hobby_data = [line.strip() for line in hobby_data]

with open("housework_data.txt",'r', encoding='utf-8') as f :
    housework_data = f.readlines()
housework_data = [line.strip() for line in housework_data]

with open("meeting_data.txt",'r', encoding='utf-8') as f :
    meeting_data = f.readlines()
meeting_data = [line.strip() for line in meeting_data]

with open("study_data.txt",'r', encoding='utf-8') as f :
    study_data = f.readlines()
study_data = [line.strip() for line in study_data]

hobby_size = len(hobby_data)
housework_size = len(housework_data)
meeting_size = len(meeting_data)
study_size = len(study_data)

hobby = ['취미' for __ in range(hobby_size)]
housework = ['가사' for __ in range(housework_size)]
meeting = ['모임' for __ in range(meeting_size)]
study = ['공부' for __ in range(study_size)]

hobby_labeled = [(a,b) for a,b in zip(hobby_data, hobby)]
housework_labeled = [(a,b) for a,b in zip(housework_data, housework)]
meeting_labeled = [(a,b) for a,b in zip(meeting_data, meeting)]
study_labeled = [(a,b) for a,b in zip(study_data, study)]

labeled_data = hobby_labeled + housework_labeled + meeting_labeled + study_labeled

data = []
label = []

for d in labeled_data :
    data.append(d[0])
    label.append(d[1])

import pandas as pd

df = pd.DataFrame({'text' : data,
                  'label' : label})

label_map = {'취미': 0, '가사': 1, '모임': 2, '공부': 3}
df['label'] = df['label'].map(label_map)

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('kykim/bert-kor-base')

#토크나이저
def Bert_Tokenizer(text_list, maximum_length = 16) :
    attention_masks = []
    input_ids = []

    for text in text_list :
        encoded = tokenizer.encode_plus(text, 
                                add_special_tokens=True ,
                                max_length=maximum_length,
                                truncation=True,
                                padding='max_length',
                                return_tensors='pt'
                                )
        attention_masks.append(encoded['attention_mask'])
        input_ids.append(encoded['input_ids'])

    return input_ids, attention_masks

#데이터셋 생성
class CustomDataset(Dataset) :
    def __init__(self, input_ids, attention_mask, label) :
        self.input_ids = torch.cat(input_ids, dim=0)
        self.attention_mask = torch.cat(attention_mask, dim=0)
        self.label = torch.tensor(label.values, dtype=torch.long)
    
    def __len__(self) :
        return len(self.label)
    
    def __getitem__(self, idx) :
        return {"input_ids" : self.input_ids[idx],
                "attention_mask" : self.attention_mask[idx],
                "label" : self.label[idx]}


#모델
class BertClassifier(nn.Module) :
    def __init__(self, bert_model, hidden_size, classes) :
        super(BertClassifier, self).__init__()

        self.bert = bert_model
        self.hidden_size = hidden_size
        self.classes = classes

        self.fc1 = nn.Linear(self.bert.config.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.hidden_size, self.classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask) :
        output = self.bert(input_ids = input_ids, attention_mask = attention_mask)
        cls = output.pooler_output
        
        x = self.fc1(cls)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
    
bert_model = BertModel.from_pretrained('kykim/bert-kor-base')
model = BertClassifier(bert_model, 64, 4)

df = df.sample(frac=1)

val_size = int(len(df) * 0.2)

train_text = df['text'][val_size:]
val_text = df['text'][:val_size]

train_label = df['label'][val_size:]
val_label = df['label'][:val_size]

train_input_ids, train_attention_mask = Bert_Tokenizer(train_text)
val_input_ids, val_attention_mask = Bert_Tokenizer(val_text)

train_dataset = CustomDataset(train_input_ids, train_attention_mask, train_label)
val_dataset = CustomDataset(val_input_ids, val_attention_mask, val_label)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=0.00001) # lr설정 매우 중요!

loss_f = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.train()

for epoch in range(3) :
    for batch in train_dataloader :
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        output = model(input_ids, attention_mask)

        loss = loss_f(output, label)
        
        loss.backward()

        optimizer.step()

model.eval()

import torch.nn.functional as F
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    data = request.json
    ids, attention_mask = Bert_Tokenizer([data])
    with torch.no_grad() :
        output = model(ids[0], attention_mask[0])
        pred_label = F.softmax(output).argmax().item()
    reverse_label_map = {0 : '취미', 1 : '가사', 2 : '모임', 3 : '공부'}
    category = reverse_label_map[pred_label]

    return jsonify({"result": category})


app.run(host="0.0.0.0", port=5001, debug=False)
