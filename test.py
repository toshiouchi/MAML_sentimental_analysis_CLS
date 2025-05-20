import torch
import torchvision
import numpy as np
import csv
from torchvision import datasets, transforms
import random
import time
from maml import MAML
from train import adaptation
import pickle
from torch.utils.data import Dataset
from collections import OrderedDict
import torch.nn.functional as F
from build_task_dataset import build_task_dataset, create_batch_of_tasks
import os
import json
from transformers import BertModel, BertTokenizer
    
def main():

    lr_inner = 0.000001

    os.makedirs( "model/", exist_ok=True)
    os.makedirs( "log/", exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    model = MAML().to(device)
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    if torch.cuda.is_available():
        checkpoint = torch.load("model/model.pth")
    else:
        checkpoint = torch.load("model/model.pth", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    result_path = "./log/test"

    # dataset

    reviews = json.load(open('dataset.json'))
    
    random.shuffle(reviews)    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    outer_optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

    for i, r in enumerate( reviews ):
        reviews[i]['text'] = tokenizer.encode( r['text'] )

    #print( "after tokenizer", reviews[:5] )
    point = len(reviews) * 4 //5
    
    testset = reviews[point:]
    
    print( "len of testset:", len( testset ) )

    num_all_class = 22
    num_task = 10
    num_class = 2 # N-way K-shot の N

    outer_batch0 = 15

    ob_test = []
    # test 用の taskset を作り outer_batch の次元を加える。
    for i in range( outer_batch0 ):
        #print( "i:", i )
        test = build_task_dataset( testset, num_all_class = num_all_class, num_task = num_task, k_support=5, k_query=5, num_class = num_class, is_val = True )
        ob_test.append( test )

    # test
    db_test = create_batch_of_tasks(ob_test, is_shuffle = False, outer_batch_size = 10)
    acc_all_test = []
    loss_all_test = []

    f = open('log.txt', 'a')
    for loop, test_task in enumerate( db_test ):
        #random_seed(123)
        #loss, acc = test_model(model, test_task, loss_fn, train_step = 10, device=device, lr1 = lr_inner)
        #loss, acc = test_model(model, test_task, loss_fn, train_step = 10, device=device, lr1 = lr_inner)
        loss, acc = adaptation(model, outer_optimizer, test_task, loss_fn,  train_step=10, train=False, device=device, lr1 = lr_inner)

        acc_all_test.append(acc)
        loss_all_test.append( loss )
        #random_seed(int(time.time() % 10))

        print('loop:', loop, 'Test loss:', np.mean(loss_all_test),'\tacc:', np.mean(acc_all_test))
        #test_loss_log.append( np.mean(loss_all_test) )
        #test_acc_log.append( np.mean(acc_all_test ) )
        f.write('Test' + str(np.mean(acc_all_test)) + '\n')
            
        
    all_result = { 'test_loss': loss_all_test, 'test_acc': acc_all_test}

    with open(result_path + '.pkl', 'wb') as f:
        pickle.dump(all_result, f)
    
if __name__ == "__main__":
    main()

