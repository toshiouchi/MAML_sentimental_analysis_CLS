import torch
import torchvision
import numpy as np
import csv
from torchvision import datasets, transforms
import random
import time
import pickle
from torch.utils.data import Dataset
from collections import OrderedDict
import torch.nn.functional as F
from collections import Counter


# 乱数の種を設定
def random_seed(value):
    torch.backends.cudnn.deterministic=True
    torch.manual_seed(value)
    torch.cuda.manual_seed(value)
    np.random.seed(value)
    random.seed(value)

# list からランダムに kosuu の数値をとってきて list を作る。
def select( list, kosuu ):
    output = random.sample( list.tolist(), kosuu )
    return output
#def select( list, kosuu ):
#    while True:
#        idx = torch.randint( 0, len(list), (kosuu,))
#        if kosuu == len( torch.unique(idx)):
#            break
#    #print( "idx:", idx )
#    return list[idx]

# ( outer_batch の次元のある)taskset から outer_batch_size の outer_batch データを作る。    
def create_batch_of_tasks(taskset, is_shuffle = True, outer_batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    idxs = torch.tensor( idxs )
    idxss = select( idxs, outer_batch_size )
    output = []

    for i in idxss:
        output.append( taskset[i])
    
    return output 

# [num_task, inner_batch, num_class * k, max_seq] テキストを tokenizer でエンコードしたタスクデータ
# [num_task, inter_batch, num_class * k ]　ラベルタスクデータを作る。
def build_task_dataset( dataset, num_all_class, num_task, k_support, k_query, num_class, inner_batch, is_val = False ):
    
    #画像データとラベルデータをシャフル    
    max_seq = 128

    examples = dataset
    random.shuffle(examples)

    text = torch.zeros( ( len(examples), max_seq ) ).long()
    text_length = torch.zeros( ( len(examples ) ) ).long()
    label = torch.zeros( ( len( examples ) ) ).long()
    domain = torch.zeros( ( len( examples ) ) ).long()
    
    mention_domain = [r['domain'] for r in examples]
    unique_domain = Counter(mention_domain).keys()
    
    domain_dict = {}
    for i, ud in enumerate( unique_domain ):
        domain_dict[ud] = i
    
    for i, r in enumerate( examples ):
        text_length[i] = len( r['text'] )
        if text_length[i] >= max_seq:
            text_length[i] = max_seq
        if r['label'] == 'positive':
            label[i] = 1
        else:
            label[i] = 0
        domain[i] = domain_dict[r['domain']]
        text[i,:text_length[i]] = torch.tensor( r['text'][:text_length[i]] )

    supports_text = []  # support text set
    supports_label = []  # support label set
    queries_text = []  # query text set
    queries_label = []  # query label set
        
    exam_support3_text = []
    exam_support3_label = []
    exam_query3_text = []
    exam_query3_label = []
    for b in range(num_task):  # タスクのループ
        # タスクは 0 ～ 21 の22個のうちランダムに選択
        if is_val == False:
            current_task_support = torch.randint( 0, num_all_class - 3, size=(1,))
        else:
            current_task_support = torch.randint( num_all_class - 3, num_all_class, size=(1,))
        #current_task_query = torch.randint( 0, num_all_class, size=(1,))
        # support データセットと query データセットのタスクは同じで良いようです。
        current_task_query = current_task_support
        exam_support2_text = []
        exam_support2_label = []
        exam_query2_text = []
        exam_query2_label = []
        for c in range( inner_batch ): # inner_batch のループ
            exam_support1_text = []
            exam_support1_label = []
            exam_query1_text = []
            exam_query1_label = []
            for k in range( num_class ): # N-way K-shot のループ N 部分
                # タスク3 だったら、 ドメインが 3 に等しい添え字を True False で取得。
                idx_support = domain ==  current_task_support
                idx_query = domain ==  current_task_query
                # 0 ～ データ数の数字を生成。
                idx2 = torch.arange( 0, len(text) , 1 ).long()
                # 上で True False の添え字を番号添え字に変換
                idx3_support = idx2[idx_support]
                idx3_query = idx2[idx_query]
                # 番号添え字の中からランダムに k_support あるいは k_query 個選ぶ。
                idx4_support = select( idx3_support, k_support )
                idx4_query = select( idx3_query, k_query )
                # 選択したデータを追加。
                for l in range( k_support ): # suppor データの N-way K-shot のループ k 部分
                    exam_support1_text.append(text[idx4_support[l]])
                    exam_support1_label.append( label[idx4_support[l]] )
                for l in range( k_query ): # query データの N-way K-shot のループ k 部分
                    exam_query1_text.append(text[idx4_query[l]] )
                    exam_query1_label.append( label[idx4_query[l]] )
            exam_support1_text = torch.stack( exam_support1_text, dim = 0 )
            exam_support1_label = torch.tensor( exam_support1_label )
            exam_query1_text = torch.stack( exam_query1_text, dim = 0 )
            exam_query1_label = torch.tensor( exam_query1_label )

            exam_support2_text.append( exam_support1_text )
            exam_support2_label.append( exam_support1_label )
            exam_query2_text.append( exam_query1_text )
            exam_query2_label.append( exam_query1_label )
        
        exam_support2_text = torch.stack( exam_support2_text, dim = 0 )
        exam_support2_label = torch.stack( exam_support2_label, dim = 0 )
        exam_query2_text = torch.stack( exam_query2_text, dim = 0 )
        exam_query2_label = torch.stack( exam_query2_label, dim = 0 )

        exam_support3_text.append( exam_support2_text )
        exam_support3_label.append( exam_support2_label )
        exam_query3_text.append( exam_query2_text )
        exam_query3_label.append( exam_query2_label )
        
    exam_support3_text = torch.stack( exam_support3_text, dim = 0 )
    exam_support3_label = torch.stack( exam_support3_label, dim = 0 )
    exam_query3_text = torch.stack( exam_query3_text, dim = 0 )
    exam_query3_label = torch.stack( exam_query3_label, dim = 0 )

    supports_text = exam_support3_text
    supports_label = exam_support3_label
    queries_text = exam_query3_text
    queries_label = exam_query3_label
    
    # アテンションマスクを作る。
    supports_attn = torch.eq( supports_text,  0 )
    queries_attn = torch.eq( queries_text, 0 )

    return supports_text, supports_attn, supports_label, queries_text, queries_attn, queries_label
