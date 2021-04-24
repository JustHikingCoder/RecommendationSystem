# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import evall
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

train_data = np.array(pd.read_csv(r'data/train_data.csv',usecols=['userId','itemId','gerne']))

#neg_data = np.array(pd.read_csv(r'data/neg_data.csv',usecols=['userId','itemId','gerne']))

test_data =pd.read_csv(r'data/test_data.csv',usecols=['itemId','gerne'])
test_data.drop_duplicates( keep='first', inplace=True) 
test_data= np.array(test_data)
user_emb_matrix = np.array(pd.read_csv(r'util/user_emb.csv',header=None)) 
user_attribute_matrix = np.array(pd.read_csv(r'util/user_attribute.csv',header=None)) 
ui_matrix = np.array(pd.read_csv(r'util/ui_matrix.csv',header=None).astype(np.int32))

test_item = np.array(pd.read_csv('test_item.csv',header =None).astype(np.int32)   )
test_attribute = np.array( pd.read_csv( 'test_attribute.csv',header =None).astype(np.int32) )

def get_testdata():
    return test_item,test_attribute

#
def get_intersection_similar_user(G_user, k):
    #G_user = np.round(G_user)
#    G_user = np.around(G_user)
    user_emb_matrixT = np.transpose(user_attribute_matrix)    
    A = np.matmul(G_user, user_emb_matrixT)   
    intersection_rank_matrix = np.argsort(-A)
#    print( intersection_rank_matrix[0,:k])
#    print( intersection_rank_matrix[50,:k])    
    return intersection_rank_matrix[:, 0:k]

def test(test_item_batch, test_G_user):

    # print("-----test_G_user-----before------",test_G_user)
    # test_item_batch = test_item_batch.detach().numpy().astype(np.int32)
    # test_G_user = test_G_user.detach().numpy().astype(np.int32)
    
    # print("-----test_item_batch------",test_item_batch)
    # print("-----test_G_user-----latter------",test_G_user)

    k_value = 20
    test_BATCH_SIZE = np.size(test_item_batch)
    
    test_intersection_similar_user = get_intersection_similar_user(test_G_user, k_value).astype(np.int32)
    
    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):       
        for test_u in test_userlist:
            
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1            
    p_at_20 = round(count/(test_BATCH_SIZE * k_value), 4)
           
    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        RS.append( r)
#    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) )  
    M_at_20 = evall.mean_average_precision(RS)
  
    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
#    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE) 
    G_at_20 = ans/test_BATCH_SIZE
    k_value = 10 
    
    count = 0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):       
        for test_u in test_userlist[:k_value]:
            
            if ui_matrix[test_u, test_i] == 1:
                count = count + 1            
    p_at_10 = round(count/(test_BATCH_SIZE * k_value), 4)
         
    ans = 0.0
    RS = []
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        RS.append( r)
#    print('MAP @ ',k_value,' is ',  evall.mean_average_precision(RS) ) 
    M_at_10 = evall.mean_average_precision(RS)
    

    ans = 0.0
    for test_i, test_userlist in zip(test_item_batch, test_intersection_similar_user):  
        r=[]
        for user in test_userlist[:k_value]:
            r.append(ui_matrix[user][test_i])
        ans = ans + evall.ndcg_at_k(r, k_value, method=1)
#    print('ndcg @ ',k_value,' is ', ans/test_BATCH_SIZE) 
    G_at_10 = ans/test_BATCH_SIZE
  

    return p_at_10,p_at_20,M_at_10,M_at_20,G_at_10,G_at_20


train = np.array(pd.read_csv('train_data.csv',header =None))
def shuffle():
    np.random.shuffle(train)
    
def get_traindata(start_index, end_index):
    
    '''get train samples'''
    batch_data = train[start_index: end_index]
    
#    print(  batch_data)
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    attr_batch = [int(i) for line in attr_batch for i in line]
    real_user_emb_batch = user_emb_matrix[user_batch]
    
    return user_batch,item_batch,attr_batch,real_user_emb_batch


neg = np.array(pd.read_csv('neg_data.csv',header =None))
def shuffle2():
    np.random.shuffle(neg)
    
def get_negdata(start_index, end_index):
    
    '''get negative samples'''
    batch_data = neg[start_index: end_index]
    
#    print(  batch_data)
    user_batch = [x[0] for x in batch_data]
    item_batch = [x[1] for x in batch_data]
    attr_batch = [x[2][1:-1].split() for x in batch_data]
    attr_batch = [int(i) for line in attr_batch for i in line]
    real_user_emb_batch = user_emb_matrix[user_batch]
   
    
    return user_batch,item_batch,attr_batch,real_user_emb_batch


def construt_negativedata():
    '''  这个方法是用来构建negative数据的 构建完了 就用不到了  生成的是  train_user_item.csv  and  test_attribute.csv'''
    print('  this is construct _negative data ')
    for i in neg_data:
        i[2]=i[2][1:-1]

#    user_batch = [x[0] for x in test_data]
#    item_batch =
##    attribute=[]
    for i in neg_data:       
        tmp=np.linspace (0,34,18)
        li = np.int32(i[2].split(','))
#        print(li)
        for j in li:
            tmp[j]=tmp[j]+1
        i[2] = np.array(tmp,dtype = np.int32)
    print( neg_data)
    neg = pd.DataFrame(neg_data)
    neg.to_csv( 'neg_data.csv',header=None,index = 0)

def construt_traindata():
    '''  这个方法是用来构建train数据的 构建完了 就用不到了  生成的是  train_user_item.csv  '''
    print('  this is construct _traindata ')
    for i in train_data:
        i[2]=i[2][1:-1]

#    user_batch = [x[0] for x in test_data]
#    item_batch =
##    attribute=[]
    for i in train_data:       
        tmp=np.linspace (0,34,18)
        li = np.int32(i[2].split(','))
#        print(li)
        for j in li:
            tmp[j]=tmp[j]+1
        i[2] = np.array(tmp,dtype = np.int32)
    print( train_data)
    train = pd.DataFrame(train_data)
    train.to_csv( 'train_data.csv',header=None,index = 0)


def construt_testdata():  
    '''  这个方法是用来构建test数据的 构建完了 就用不到了  生成的是 test_item.csv   and  test_attribute.csv'''
    
    for i in test_data:
        i[1]=i[1][1:-1]
    print(test_data)
    item_batch = [x[0] for x in test_data]
    attribute=[]
    for i in test_data:       
        tmp=np.linspace (0,34,18)
        li = np.int32(i[1].split(','))
        print(li)
        for j in li:
            tmp[j]=tmp[j]+1
        attribute.append(tmp)
    print(len(item_batch))
    print(len(attribute))
    item =  pd.DataFrame(item_batch )
    item.to_csv( 'test_item.csv',header=None,index = 0)
    attribute=pd.DataFrame(attribute )
    attribute.to_csv( 'test_attribute.csv',header=None,index = 0)
    
def control():
    construt_testdata()
    construt_traindata()
    construt_negativedata()
    




