import json
from utils import *
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup, BertConfig
import torch.multiprocessing as mp
import numpy as np
import random
import os, sys
from accelerate import Accelerator
from make_logger import CreateLogger
import argparse
import datetime
from torch import nn
from load_BERTdata import *
from tqdm import tqdm
from sklearn import metrics

class PadCollate():
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def pad_collate(self, batch):
        abst, label, mask = [], [], []
        for idx, seqs in enumerate(batch):
            abst.append(torch.LongTensor(seqs[0]))
            label.append(torch.LongTensor([seqs[1]]))
            mask.append(torch.LongTensor(seqs[2]))
        abst = torch.nn.utils.rnn.pad_sequence(abst, batch_first=True, padding_value=self.pad_id)
        label = torch.cat(label, dim=0)
        mask = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
        return abst,  label, mask
    
        
class PDBert(nn.Module):
    def __init__(self, args, logger):
        super(PDBert, self).__init__()
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.max_len = args.max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_length=self.max_len)
        self.pad_id = self.tokenizer.pad_token_id
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, max_length=self.max_len)
        self.model.to(self.device)
        self.logger = logger
        self.is_main = self.accelerator.is_main_process
        self.best_loss = 1e10
        self.last_epoch = 0
        

        train_set = get_dataset(self.tokenizer, 'train', args)
        test_set = get_dataset(self.tokenizer, 'test', args)

        self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.train_batch_size, shuffle=True, collate_fn=PadCollate(self.pad_id).pad_collate)
        self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=True, collate_fn=PadCollate(self.pad_id).pad_collate)

        ckpt_path = f"{self.args.output_dir}/{self.args.ckpt_name}.ckpt"
        if os.path.exists(ckpt_path):
            if self.is_main:
                self.logger.info(f"Loading checkpoint from {self.args.output_dir}/{self.args.ckpt_name}.ckpt")
            ckpt = torch.load(ckpt, map_location=self.device)
            pre_trained = ckpt['model_state_dict']
            self.best_loss = ckpt['best_loss']
            self.last_epoch = ckpt['epoch']
            new_model_dict = self.model.state_dict()
            pretrained = {k: v for k, v in pre_trained.items() if k in new_model_dict}
            if len(pretrained) == 0:
                pretrained = {k.replace('module.',''): v for k, v in pre_trained.items() if k.replace('module.','') in new_model_dict.keys()}
            if len(pretrained) == 0:
                logger.info("No matching keys found in checkpoint. Doing nothing.")
                exit()
            new_model_dict.update(pre_trained)
            self.model.load_state_dict(new_model_dict)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0, num_training_steps=len(self.train_loader)*args.epochs)
        self.criterion = nn.CrossEntropyLoss()
        self.model, self.optimizer, self.train_loader, self.test_loader = self.accelerator.prepare(self.model, self.optimizer, self.train_loader, self.test_loader)            
            
    def running_train(self):  
        self.set_seed(self.args.seed)
        if self.is_main:
            logger.info("***** Running training *****")
        start_epoch = self.last_epoch
        for epoch in range(start_epoch, self.args.epochs+start_epoch):
            if self.is_main:
                txt = f"#"*50 + f"Epoch: {epoch}" + "#"*50
                logger.info(txt)
            train_losses = []
            acc, precision, recall, f1 = [], [], [], []
            self.model.train()
            for i, batch in enumerate(tqdm(self.train_loader, desc = 'Training', disable = not self.is_main)):
                with self.accelerator.accumulate(self.model):
                    abst,  label, mask = batch
                    abst,  label, mask = abst.to(self.device),label.to(self.device), mask.to(self.device)
                    outputs = self.model(abst, labels=label, attention_mask=mask)
                    loss = self.criterion(outputs['logits'], label)
                    train_losses.append(loss.item())
                    self.optimizer.zero_grad()
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    preds = torch.argmax(outputs['logits'], dim=1)
                    acc.append(metrics.accuracy_score(label.cpu(), preds.cpu()))
                    precision.append(metrics.precision_score(label.cpu(), preds.cpu()))
                    recall.append(metrics.recall_score(label.cpu(), preds.cpu()))
                    f1.append(metrics.f1_score(label.cpu(), preds.cpu()))

            if self.is_main:
                logger.info(f"Train Loss: {np.mean(train_losses)}")
                logger.info(f"Train Accuracy: {np.mean(acc)}")
                logger.info(f"Train Precision: {np.mean(precision)}")
                logger.info(f"Train Recall: {np.mean(recall)}")
                logger.info(f"Train F1: {np.mean(f1)}")

            valid_loss, valid_acc, valid_precision, valid_recall, valid_f1 = self.valid()
            if self.is_main:
                logger.info(f"Valid Loss: {valid_loss}")
                logger.info(f"Valid Accuracy: {valid_acc}")
                logger.info(f"Valid Precision: {valid_precision}")
                logger.info(f"Valid Recall: {valid_recall}")
                logger.info(f"Valid F1: {valid_f1}")

            
            self.accelerator.wait_for_everyone()
            if self.is_main:
                if valid_loss < self.best_loss:
                    self.best_loss = valid_loss
                stated_dict = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_loss': self.best_loss,
                    'epoch': epoch,
                    'scheduler': self.scheduler.state_dict()
                }
                logger.info(f"***** Best model saved at {self.args.output_dir}/{self.args.ckpt_name}.ckpt *****")
                torch.save(stated_dict, f"{self.args.output_dir}/best_epoch{epoch}_valid_loss={round(valid_loss, 4)}.ckpt")
                logger.info(f"Best valid loss: {self.best_loss}")
                logger.info(f"Current valid loss: {valid_loss}")
                logger.info(f"Best valid accuracy: {valid_acc}")
                logger.info(f"Best valid precision: {valid_precision}")
                logger.info(f"Best valid recall: {valid_recall}")
                logger.info(f"Best valid f1: {valid_f1}")



    def valid(self):
        if self.is_main:
            logger.info("***** Running validation *****")
        self.model.eval()
        valid_losses = []
        acc, precision, recall, f1 = [], [], [], []

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader, desc = 'Validating...', disable = not self.is_main)):
                abst,  label, mask = batch
                abst,  label, mask = abst.to(self.device), label.to(self.device), mask.to(self.device)
                outputs = self.model(abst,  labels=label, attention_mask=mask)
                loss = self.criterion(outputs['logits'], label)
                valid_losses.append(loss.item())
                preds = torch.argmax(outputs['logits'], dim=1)
                acc.append(metrics.accuracy_score(label.cpu(), preds.cpu()))
                precision.append(metrics.precision_score(label.cpu(), preds.cpu()))
                recall.append(metrics.recall_score(label.cpu(), preds.cpu()))
                f1.append(metrics.f1_score(label.cpu(), preds.cpu()))

        return np.mean(valid_losses), np.mean(acc), np.mean(precision), np.mean(recall), np.mean(f1)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed_all(self.args.seed) # if use multi-GPU
                    
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--log_name', type=str, default='PDBert')
    parser.add_argument('--ckpt_name', type=str, default='PDBert')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16)

    args = parser.parse_args()
    date_txt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logger = CreateLogger('PDBert', f'./output/logs/{args.log_name}{date_txt}.log')
    NGPU = torch.cuda.device_count()
    args.n_gpus = NGPU
    args.local_rank = [i for i in range(NGPU)]
    args.train_batch_size = int(args.train_batch_size / args.n_gpus)
    args.eval_batch_size = int(args.eval_batch_size / args.n_gpus)
    args.gradient_accumulation_steps = int(args.gradient_accumulation_steps / args.n_gpus)
    args.lr = args.lr * args.n_gpus

    if args.n_gpus > 1:
        mp.set_start_method('spawn')

    model = PDBert(args, logger)
    if args.mode == 'train':
        model.running_train()