from torchvision import transforms
from torch.utils.data import  DataLoader
from crnn_dataloader import STRDataset
from utils import encode,build_vocab,split_train_test_val,fit
from opts import parse_opts_offline
from crnn import CRNN
import torch
from torch import nn
import random
import os
import numpy as np
data_transforms = {
    'train':transforms.Compose([
        transforms.Resize((100, 420)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5), 
        transforms.Grayscale(num_output_channels=1), 
        transforms.GaussianBlur(3),
        transforms.RandomAffine(degrees=1, shear=1), 
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5,interpolation=3),
        transforms.RandomRotation(degrees=2), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)),
    ]),
    'val': transforms.Compose([
        transforms.Resize((100, 420)), 
        transforms.Grayscale(num_output_channels=1), 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))])
}
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
if __name__ == "__main__":
    seed_everything(20)
    opts = parse_opts_offline()
    labels,X_train,y_train,X_val,y_val,X_test,y_test = split_train_test_val(opts)
    vocab_size,char_to_idx,idx_to_char,max_label_len = build_vocab(labels)
    train_dataset = STRDataset(
        X_train , y_train , char_to_idx=char_to_idx , max_label_len=max_label_len , 
        label_encoder=encode , transform=data_transforms['train'])
    val_dataset = STRDataset(
        X_val , y_val , char_to_idx=char_to_idx , max_label_len=max_label_len , 
        label_encoder=encode , transform=data_transforms['val'])
    test_dataset = STRDataset(
        X_test , y_test , char_to_idx=char_to_idx , max_label_len=max_label_len , 
        label_encoder=encode , transform=data_transforms['val']
    )
    train_loader = DataLoader( train_dataset ,
        batch_size=opts.train_batch_size ,num_workers=3,
        shuffle=True )
    val_loader = DataLoader( val_dataset ,num_workers=3,
        batch_size=opts.val_batch_size ,
        shuffle=False )
    test_loader = DataLoader( test_dataset ,num_workers=3,
        batch_size=opts.test_batch_size ,
        shuffle=False )
    
    model = CRNN(vocab_size=vocab_size , hidden_size=opts.hidden_size , n_layers=opts.n_layers , 
                 dropout=opts.dropout_prob , unfreeze_layers=opts.unfreeze_layers).to(opts.device)
    epochs = 35
    lr = 0.001
    weight_decay=1e-5 
    scheduler_step_size = epochs * 0.6
    criterion = nn.CTCLoss(blank=char_to_idx['-'], zero_infinity=True) 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer , step_size=scheduler_step_size ,gamma =0.1 )
    model  = fit( model ,train_loader , val_loader , criterion , optimizer , scheduler ,opts.device , epochs)