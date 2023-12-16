import torch
import os
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from tqdm.auto import tqdm

def read_data(opts):
    img_paths = []
    labels = []
    # Read labels from text file
    with open(os.path.join(opts.crnn_data_dir , 'labels.txt'), 'r') as f:
        for label in f:
            labels.append(label.strip().split("\t")[1])
            img_paths.append(label.strip().split("\t")[0])
            
    print(f"Total images: {len(img_paths)}")
    return labels,img_paths
    
def build_vocab(labels):
    letters = [char.split(".")[0].lower() for char in labels]
    letters = "".join(letters)
    letters = sorted(list(set(list(letters))))
    chars = "".join(letters)
    # for "blank" character
    blank_char = '-'
    chars += blank_char 
    vocab_size = len(chars)
    print(f'Vocab size: {vocab_size}')
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))} 
    idx_to_char = {index: char for char, index in char_to_idx.items()}
    max_label_len = max([len(label) for label in labels])
    return vocab_size,char_to_idx,idx_to_char,max_label_len
    
def encode(label,char_to_idx,max_label_len):
    encoded_labels = torch.tensor(
            [char_to_idx[char] for char in label],dtype=torch.long )
    label_len = len(encoded_labels)
    lengths = torch.tensor(label_len ,dtype=torch.long )
    padded_labels = F.pad( encoded_labels ,(0, max_label_len - label_len),value =0 )
    return padded_labels , lengths

def decode(encoded_sequences , idx_to_char , blank_char='-'): 
    decoded_sequences = []
    for seq in encoded_sequences: 
        decoded_label = []
        blank_idx = 0
        for idx, token in enumerate(seq):
            if token != 0:
                char = idx_to_char[token.item()] 
                if char != blank_char:
                    decoded_label.append(char)
                else:
                    cpy = decoded_label[blank_idx:len(decoded_label)].copy()
                    a = []
                    for i in cpy:
                        if i not in a:
                            a.append(i)
                    decoded_label = decoded_label[:blank_idx] + a
                    blank_idx = len(decoded_label)
                        
                    
        decoded_sequences.append(''.join(decoded_label))
    return decoded_sequences

def split_train_test_val(opts,seed = 0,val_size = 0.1,test_size = 0.1,is_shuffle = True):
    labels,img_paths = read_data(opts)
    X_train, X_val, y_train, y_val = train_test_split( 
                img_paths , labels ,
                test_size=val_size ,
                random_state=seed,
                shuffle=is_shuffle )
    X_train, X_test, y_train, y_test = train_test_split( 
                X_train , y_train ,
                test_size=test_size ,
                random_state=seed,
                shuffle=is_shuffle )
    
    return labels,X_train,y_train,X_val,y_val,X_test,y_test




def evaluate(model , dataloader , criterion , device): 
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, labels, labels_len in dataloader: 
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_len = labels_len.to(device)
            outputs = model(inputs) 
            logits_lens = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs , labels , logits_lens , labels_len)
            losses.append(loss.item()) 
    loss = sum(losses) / len(losses)
    return loss

def fit( model ,train_loader , val_loader , criterion , optimizer , scheduler , device , epochs):  
   
    val_score = pd.DataFrame(columns=["Losss"])
    train_score = pd.DataFrame(columns=["Losss"])
    train_dir = 'models/rcnn/train'
    val_dir = 'models/rcnn/val'
    save_model = "models/model"
    best_loss_eval = 100
    os.makedirs(train_dir,exist_ok=True)
    os.makedirs(val_dir,exist_ok=True)
    os.makedirs(save_model,exist_ok=True)
    for epoch in range(epochs):
        batch_train_losses = []
        model.train()
        for idx, (inputs, labels, labels_len) in enumerate(tqdm(train_loader)):
            inputs = inputs.to(device)
            labels = labels.to(device) 
            labels_len = labels_len.to(device)
            optimizer.zero_grad() 
            outputs = model(inputs)
            logits_lens = torch.full( size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs , labels , logits_lens , labels_len) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5 )
            optimizer.step()
            batch_train_losses.append(loss.item())
        train_score.loc[len(train_score)] =  sum(batch_train_losses) / len(batch_train_losses)
        eval_loss = evaluate( model , val_loader , criterion , device)
        val_score.loc[len(val_score)] = eval_loss
        train_score.to_csv(train_dir + "/train_score.csv")
        val_score.to_csv(val_dir + "/val_score.csv")
        if eval_loss < best_loss_eval:
            torch.save(model.state_dict(), save_model + f'/crnn.pt')
            best_loss_eval = eval_loss
    model.load_state_dict(torch.load(save_model + f'/crnn.pt'))
    model.eval()
    return model 
