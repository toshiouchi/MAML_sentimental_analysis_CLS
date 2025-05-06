import torch
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np

max_seq = 128

def adaptation(model, outer_optimizer, batch, loss_fn, train_step, train, device, lr1 ):

    x_train = batch[0] #support tokeinzer.encode text データ
    a_train = batch[1] #support attention mask データ
    y_train = batch[2] #support ラベルデータ
    x_val = batch[3]   #query  tokeinzer.encode text データ
    a_val = batch[4]   #query attention mask データ
    y_val = batch[5]   #query ラベルデータ

    task_accs = []
    num_task = len( x_train )

    outer_loss_item = 0

    if train:
        weights0 = OrderedDict(model.named_parameters()) #今回の基準パラメータ
    for idx in range(x_train.size(0)): # task
        if train:
            weights = weights0
        weights2 = OrderedDict(model.named_parameters()) #今回の基準パラメータ, 
                                # for 文の中だが、weights2は更新されるが、model.parameter は変わらない。
        # batch 抽出
        input_x = x_train[idx].to(device)
        input_a = a_train[idx].to(device)
        input_y = y_train[idx].to(device)
        
        print('----Task',idx, '----')

        # タスクごとの損失の計算
        loss_item = 0
        for iter in range(train_step): # train_step のループ
            x = input_x
            a = input_a
            y = input_y
            logits = model.adaptation(x, a, weights2 )
            loss = loss_fn(logits, y )
            loss_item += loss.item()
            #各タスクについて一番目の損失関数からモデルパラメーターを求める。
            gradients = torch.autograd.grad(loss, weights2.values(), create_graph = True )
            weights2 = OrderedDict((name, param - lr1 * grad) for ((name, param), grad) in zip(weights2.items(), gradients))

        loss_item = loss_item / train_step 

        print("Inner Loss: ", loss_item)
        
        # query データからバッチ抽出
        input_x = x_val[idx].to(device)
        input_a = a_val[idx].to(device)
        input_y = y_val[idx].to(device)
        
        # 訓練時に query データ（inner_batch = 1, query_k * 2 クラス )　で二番目の損失関数の各タスクについての総和を求める。
        x = input_x.view( -1, max_seq ) # query_batch = 1
        a = input_a.view( -1, max_seq )
        y = input_y.view( -1 )  # query_batch = 1
        # 各タスクについて、上で求めたモデルパラメーターを使って損失を求める。
        if train:
            model.train()
            logits = model.adaptation( x, a, weights2 )
        else:
            model.eval()
            with torch.no_grad():
                logits = model.adaptation( x, a, weights2 )

        outer_loss = loss_fn( logits, y )
        outer_loss_item += outer_loss.item()

        pre_label_id = torch.argmax( logits, dim = 1 )
        acc = torch.sum( torch.eq( pre_label_id, y ).float() ) / y.size(0)
        task_accs.append(acc)

        if train:
            tmp = torch.autograd.grad( outer_loss, weights.values() )
            if idx ==0:
                gradients2 = list(tmp)
            else:
                #gradients2 += list(tmp)
                gradients2 = [x + y for x, y in zip(gradients2, list(tmp))]
        
    if train:
        outer_optimizer.zero_grad()
        for i, params in enumerate(model.parameters()):
            params.grad = gradients2[i]
        outer_optimizer.step()
            
    task_accs = torch.stack( task_accs )
            
    return outer_loss_item / num_task, torch.mean(task_accs).item()

def test_model(model, batch, loss_fn, train_step, device,lr1):
    #評価用ルーチン
    x_train = batch[0] #support tokeinzer.encode text データ
    a_train = batch[1] #support attention mask データ
    y_train = batch[2] #support ラベルデータ
    x_val = batch[3]   #query  tokeinzer.encode text データ
    a_val = batch[4]   #query attention mask データ
    y_val = batch[5]   #query ラベルデータ

    
    predictions = []
    labels = []

    loss1 = 0
    loss2 = 0

    task_accs = []

    for idx in range(x_train.size(0)): # task
        # model.parameter を weights 関数に格納。 idx のループの間、weights は書き換えられるが model.parameter は変わらない。
        weights = OrderedDict(model.named_parameters()) #今回の基準パラメータ
        # batch 抽出
        input_x = x_train[idx].to(device)
        input_a = a_train[idx].to(device)
        input_y = y_train[idx].to(device)
        
        print('----Task',idx, '----')

        # 各タスクについて train_step 回学習をループし、パラメーターを求める。
        for iter in range(train_step):
            x = input_x.view( -1, max_seq ) 
                # support_batch [ inner_batch, max_seq] → [ inner_batch * N * K, max_seq]
            a = input_a.view( -1, max_seq )
            y = input_y.view( -1 )  # support_batch
            logits = model.adaptation(x, a, weights)
            loss = loss_fn(logits, y)
            loss1 += loss
            gradients = torch.autograd.grad(loss, weights.values())
            #gradients = torch.autograd.grad(loss, weights.values(), create_graph=True)
            weights = OrderedDict((name, param - lr1 * grad) for ((name, param), grad) in zip(weights.items(), gradients))

        loss1 = loss1 / (iter + 1 )
        
        print( "Inner loss:", loss1.item() )

        #各タスクについて上で求めた weights を用い、損失と精度を計算する。
        with torch.no_grad():
            # query data
            input_x = x_val[idx].to(device)
            input_a = a_val[idx].to(device)
            input_y = y_val[idx].to(device)
            inner_batch = len( input_x ) # 1だと思うけど。
            x = input_x.view( -1, max_seq ) # query_batch = 1
            a = input_a.view( -1, max_seq )
            y = input_y.view( -1 )  # query_batch = 1
            logits = model.adaptation( x, a, weights )
            pred_label_id = torch.argmax( logits, dim = 1 )
            loss2 += loss_fn( logits, y )
            
            y_pred = logits.softmax( dim = 1 )
            acc = torch.sum( torch.eq( pred_label_id, y ).float() ) / y.size(0)
            print( "acc:", acc )
            task_accs.append(acc.item())
            predictions.append(y_pred)
            labels.append(y)

    y_pred = torch.cat(predictions)
    y_label = torch.cat(labels)
    batch_acc = torch.eq(y_pred.argmax(dim=-1), y_label).sum().item() / y_pred.shape[0]            

    #タスクについての平均の損失と、すべてのタスクで計算した精度を表示する。
    print( "loss2:", loss2.item() / ( idx + 1 ) / inner_batch )
    print( "batch_acc:", batch_acc )

    return loss2.item() / ( idx + 1 ) / inner_batch,  batch_acc


