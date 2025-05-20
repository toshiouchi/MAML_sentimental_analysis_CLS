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
    #print( "len taskset:", len( taskset ) ) # outer_batch
    #print( "len taskset[0]:", len( taskset[0] ) ) # support_text, suppotr_label, query_text, query_label
    #print( "len taskset[0][0]:", len( taskset[0][0] ) ) # task
    #print( "len taskset[0][0][0]:", len( taskset[0][0][0] ) ) # innter_bath
    #print( "len taskset[0][0][0][0]:", len( taskset[0][0][0][0] ) ) # class * k
    #print( "len taskset[0][0][0][0][0]:", len( taskset[0][0][0][0][0] ) ) #text

    output = []
    output0 = {}
    max_sup = 0
    max_que = 0
    #ci0 = 0
    #ci2 = 0
    for a1 in taskset: # outer_batch
        for a2 in a1[0]:  # koumoku
            for a3 in a2: # task
              if max_sup < len( a3 ) :
                  max_sup = len( a3 )
               #print( "type of a3:", type( a3 ) )
               #for a4 in a3: # inner_batch
               #   #print( "type a4:", type( a4 ) )
               #   #print( a4 )
               #   #if ci0 == 0:
               #   #    print( "0 a4:", a4 )
               #   #    ci0 += 1
               #   if max_sup < len( a4) :
               #       max_sup = len( a4 )
        for a2 in a1[2]:
            for a3 in a2:
              if max_que < len( a3 ):
                  max_que = len( a3 )
               #for a4 in a3:
               #   #if ci2 == 0:
               #   #    print( "2 a4:", a4 )
               #   #    ci2 += 1
               #   if max_que < len( a4 ):
               #       max_que = len( a4 )

    print( "max_sup:", max_sup )
    print( "max_que:", max_que )

    #torch_tensor_sup_text = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ), len( taskset[0][0][0][0] ), max_sup ).long()
    #torch_tensor_sup_label = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ), len( taskset[0][0][0][0] ) ).long()
    ##print( "size of torch_tensor_sup:", torch_tensor_sup.size() )
    ##exit()
    #torch_tensor_que_text = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ), len( taskset[0][0][0][0] ), max_que ).long()
    #torch_tensor_que_label = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ), len( taskset[0][0][0][0] ) ).long()
    torch_tensor_sup_text = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ), max_sup ).long()
    torch_tensor_sup_label = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ) ).long()
    #print( "size of torch_tensor_sup:", torch_tensor_sup.size() )
    #exit()
    torch_tensor_que_text = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ), max_que ).long()
    torch_tensor_que_label = torch.zeros( len( taskset ), len( taskset[0][0]), len( taskset[0][0][0] ) ).long()


    #print( "size of torch_tensor_query:", torch_tensor_que_text.size() )

    for i, a1 in enumerate( taskset ): # outer_batch
        for j, a2 in enumerate( a1[0]): # task, 0, support_text, support_label, query_text, query_label
            #for k, a3 in enumerate( a2 ): #innter_batch
            #    for l, a4 in enumerate( a3 ): # class * k
            for l, a4 in enumerate( a2 ): # class * k
                    #print( "i:", i, " j:", j, " k:",k, " l:", l )
                    #print( "len taskset:", len( taskset ) )
                    #print( "len a1[0]:", len( a1[0] ) )
                    #print( "len a2:", len( a2 ) )
                    #print( "len a3:", len( a3 ) )
                    #print( "type of a4:", type( a4 ) )
                    #print( "len a4:", len( a4 ) )
                length = len( a4 )
                pad_len = max_sup - length
                torch_tensor_sup_text[i,j,l] = torch.tensor( np.pad( a4, (( 0, pad_len ) ) ) )
        for j, a2 in enumerate( a1[1]): # task, 0, support_text, support_label, query_text, query_label
            #for k, a3 in enumerate( a2 ): #innter_batch
            #    for l, a4 in enumerate( a3 ): # class * k
            for l, a4 in enumerate( a2 ): # class * k
                    #print( "len taskset:", len( taskset ) )
                    #print( "len a1[0]:", len( a1[0] ) )
                    #print( "len a2:", len( a2 ) )
                    #print( "len a3:", len( a3 ) )
                    #print( "len a4:", len( a4 ) )
                    #torch_tensor_sup_label[i,j,k,l] = a4
                torch_tensor_sup_label[i,j,l] = a4
        for j, a2 in enumerate( a1[2]): # task, 0, support_text, support_label, query_text, query_label
            #for k, a3 in enumerate( a2 ): #innter_batch
            #    for l, a4 in enumerate( a3 ): # class * k
            for l, a4 in enumerate( a2 ): # class * k
                    #print( "len taskset:", len( taskset ) )
                    #print( "len a1[0]:", len( a1[0] ) )
                    #print( "len a2:", len( a2 ) )
                    #print( "len a3:", len( a3 ) )
                    #print( "len a4:", len( a4 ) )
                length = len( a4 )
                pad_len = max_que - length
                torch_tensor_que_text[i,j,l] = torch.tensor( np.pad( a4, (( 0, pad_len ) ) ) )
        for j, a2 in enumerate( a1[3]): # task, 0, support_text, support_label, query_text, query_label
            #for k, a3 in enumerate( a2 ): #innter_batch
            #    for l, a4 in enumerate( a3 ): # class * k
            for l, a4 in enumerate( a2 ): # class * k
                    #print( "len taskset:", len( taskset ) )
                    #print( "len a1[0]:", len( a1[0] ) )
                    #print( "len a2:", len( a2 ) )
                    #print( "len a3:", len( a3 ) )
                    #print( "len a4:", len( a4 ) )
                    #torch_tensor_que_label[i,j,k,l] = a4
                torch_tensor_que_label[i,j,l] = a4

        torch_tensor_sup_attn = torch.eq( torch_tensor_sup_text, 0 )
        torch_tensor_que_attn = torch.eq( torch_tensor_que_text, 0 )


    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
        idxs = torch.tensor( idxs )
        idxss = select( idxs, outer_batch_size )
    else:
        idxs = torch.tensor( idxs )
        idxss = idxs[ :outer_batch_size ]
        
    #output_s_t = []
    #output_s_a = []
    #output_s_l = []
    #output_q_t = []
    #output_q_a = []
    #output_q_l = []
    output = []

    for i in idxss:
        line = [ torch_tensor_sup_text[i], torch_tensor_sup_attn[i], torch_tensor_sup_label[i], torch_tensor_que_text[i],
        torch_tensor_que_attn[i], torch_tensor_que_label[i] ]
        output.append( line )
        # [torch_tensor_sup_text[i])
        #output_s_a.append(torch_tensor_sup_attn[i])
        #output_s_l.append(torch_tensor_sup_label[i])
        #output_q_t.append(torch_tensor_que_text[i])
        #output_q_a.append(torch_tensor_que_attn[i])
        #output_q_l.append(torch_tensor_que_label[i])

    return output 
    #return (output_s_t, output_s_a, output_s_l, output_q_t, output_q_a, output_q_l)

# [num_task, num_class * k, max_seq] テキストを tokenizer でエンコードしたタスクデータ
# [num_task, num_class * k ]　ラベルタスクデータを作る。
def build_task_dataset( dataset, num_all_class, num_task, k_support, k_query, num_class,is_val = False ):
    
    #画像データとラベルデータをシャフル    
    max_seq = 128

    examples = dataset
    random.shuffle(examples)

    #text = torch.zeros( ( len(examples), max_seq ) ).long()
    #text_length = torch.zeros( ( len(examples ) ) ).long()
    text = []
    label = torch.zeros( ( len( examples ) ) ).long()
    domain = torch.zeros( ( len( examples ) ) ).long()
    
    mention_domain = [r['domain'] for r in examples]
    unique_domain = Counter(mention_domain).keys()
    
    domain_dict = {}
    for i, ud in enumerate( unique_domain ):
        domain_dict[ud] = i
    
    for i, r in enumerate( examples ):
        #text_length[i] = len( r['text'] )
        #if text_length[i] >= max_seq:
        #    text_length[i] = max_seq
        if r['label'] == 'positive':
            label[i] = 1
        else:
            label[i] = 0
        domain[i] = domain_dict[r['domain']]
        #text[i,:text_length[i]] = torch.tensor( r['text'][:text_length[i]] )
        #print( "type r['text']:", type( r['text'] ) )
        text.append( r['text'] )

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
        for k in range( num_class ): # N-way K-shot のループ N 部分
            # タスク3 だったら、 ドメインが 3 に等しい添え字を True False で取得。
            #idx_support = domain ==  current_task_support
            #idx_query = domain ==  current_task_query
            idx = domain== current_task_support
            # 0 ～ データ数の数字を生成。
            idx2 = torch.arange( 0, len(text) , 1 ).long()
            # 上で True False の添え字を番号添え字に変換
            #idx3_support = idx2[idx_support]
            #idx3_query = idx2[idx_query]
            idx3 = idx2[idx]
            idx4 = select( idx3, k_support + k_query )
            # 番号添え字の中からランダムに k_support あるいは k_query 個選ぶ。
            #idx4_support = select( idx3_support, k_support )
            #idx4_query = select( idx3_query, k_query )
            idx4_support = idx4[:k_support]
            idx4_query = idx4[k_support:]
            # 選択したデータを追加。
            for l in range( k_support ): # suppor データの N-way K-shot のループ k 部分
                exam_support2_text.append(text[idx4_support[l]])
                exam_support2_label.append( label[idx4_support[l]] )
            for l in range( k_query ): # query データの N-way K-shot のループ k 部分
                exam_query2_text.append(text[idx4_query[l]] )
                exam_query2_label.append( label[idx4_query[l]] )

        
        exam_support2_label = torch.stack( exam_support2_label, dim = 0 )
        exam_query2_label = torch.stack( exam_query2_label, dim = 0 )

        exam_support3_text.append( exam_support2_text )
        exam_support3_label.append( exam_support2_label )
        exam_query3_text.append( exam_query2_text )
        exam_query3_label.append( exam_query2_label )
        
    exam_support3_label = torch.stack( exam_support3_label, dim = 0 )
    exam_query3_label = torch.stack( exam_query3_label, dim = 0 )

    supports_text = exam_support3_text
    supports_label = exam_support3_label
    queries_text = exam_query3_text
    queries_label = exam_query3_label
    
    return supports_text, supports_label, queries_text, queries_label
