import torch
import torchvision
import numpy as np
import csv
import json
#from torchvision import datasets, transforms
import random
import time
from maml import MAML
from train import adaptation

import pickle
from torch.utils.data import Dataset
from build_task_dataset import build_task_dataset, create_batch_of_tasks, random_seed
import os
from transformers import BertModel, BertTokenizer

def main():

    #ディレクトリを作る。
    os.makedirs( "model/", exist_ok=True)
    os.makedirs( "log/", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    epochs = 100
    model = MAML().to(device)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    lr_inner = 0.000001
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    model_path = "./model/"
    result_path = "./log/train"
    
    #for name in model.named_parameters():
    #    print( name )

    # dataset
    reviews = json.load(open('dataset.json'))
    random.shuffle(reviews)    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)

    for i, r in enumerate( reviews ):
        reviews[i]['text'] = tokenizer.encode( r['text'] )
        
    #print( reviews[0] )

    point1 = len(reviews) * 3 // 5
    point2 = len(reviews) * 4 // 5
    
    trainset = reviews[:point1]
    valset = reviews[point1:point2]
    
    print( "len of trainset:", len( trainset ) )
    print( "len of valset:", len( valset ) )

    train_loss_log = []
    train_acc_log = []
    val_loss_log = []
    val_acc_log = []

    num_all_class = 22
    num_task = 10
    num_class = 2 # N-way K-shot の N

    print( "epochs:", epochs )

    outer_batch0 = 3 # 実際の outer_batch 数より大きめの値を設定しておく。

    ob_val = []
    # validation 用の taskset を作り outer_batch の次元を加える。
    for i in range( outer_batch0 ):
        val = build_task_dataset( valset, num_all_class = num_all_class, num_task = num_task, k_support=5, k_query=5, num_class = num_class, is_val = True )
        ob_val.append( val )

    global_step = 0

    for epoch in range(epochs):

        ob_train = []
        # 学習用の taskset を作り outer_batch の次元を加える。
        for i in range( outer_batch0 ):
            train = build_task_dataset(trainset, num_all_class = num_all_class, num_task = num_task, k_support=5, k_query=5, num_class = num_class, is_val = False )
            ob_train.append( train )

        # 学習用データセットを作る。
        db_train = create_batch_of_tasks( ob_train, is_shuffle = True, outer_batch_size = 1 )

        for step, train_task in enumerate(db_train):
        
            f = open('log.txt', 'a')
        
            #学習。
            loss, acc = adaptation(model, outer_optimizer, train_task, loss_fn,  train_step=5, train=True, device=device, lr1 = lr_inner)
            train_loss_log.append( loss )
            train_acc_log.append( acc )   
        
            print('Epoch:', epoch, 'Step:', step, '\ttraining Loss:', loss,'\ttraining Acc:', acc)
            f.write(str(acc) + '\n')
        
            # Validation
            if global_step % 20 == 0:
                random_seed(123)
                print("\n-----------------Validation Mode-----------------\n")
                db_val = create_batch_of_tasks(ob_val, is_shuffle = False, outer_batch_size = 3)
                acc_all_val = []
                loss_all_val = []

                for val_task in db_val:
                    loss, acc = adaptation(model, outer_optimizer, val_task, loss_fn,  train_step=10, train=False, device=device, lr1 = lr_inner)
                    acc_all_val.append(acc)
                    loss_all_val.append( loss )


                print('Epoch:', epoch, 'Step:', step, 'Validation F1 loss:', np.mean(loss_all_val),'\tacc:', np.mean(acc_all_val))
                val_loss_log.append( np.mean(loss_all_val) )
                val_acc_log.append( np.mean(acc_all_val ) )
                f.write('Validation' + str(np.mean(acc_all_val)) + '\n')
            
                random_seed(int(time.time() % 10))
        
            global_step += 1
            f.close()
        
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': outer_optimizer.state_dict(),
                'loss': loss,},
               model_path + 'model.pth')

    all_result = {'train_loss': train_loss_log, 'train_acc': train_acc_log, 'val_loss': val_loss_log, 'val_acc': val_acc_log}

    with open(result_path + '.pkl', 'wb') as f:
        pickle.dump(all_result, f)
    
if __name__ == "__main__":
    main()
