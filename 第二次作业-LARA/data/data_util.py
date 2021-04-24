# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

#from sklearn.model_selection import train_test_split
import random

def get_data():
    
    data = pd.read_csv('newdata.csv',usecols=['userID','itemID','rating','gerne','director','writer'])
        
    data['tmp']=data['director'].str.split(',',expand=True)[0]
    data['director']=data['tmp'].str.split('m',expand=True)[1]
    data=data.dropna()
    data['director']=data['director'].astype(np.int32)     

    data['writer0']=data['writer'].str.split(',',expand=True)[0]
    data['writer1']=data['writer'].str.split(',',expand=True)[1]
    data['writer2']=data['writer'].str.split(',',expand=True)[2]
    data['writer3']=data['writer'].str.split(',',expand=True)[3]
    
    data = data.fillna(value='nm0')
    data['writer0']=data['writer0'].str.split('m',expand=True)[1]
    data['writer1']=data['writer1'].str.split('m',expand=True)[1]
    data['writer2']=data['writer2'].str.split('m',expand=True)[1]
    data['writer3']=data['writer3'].str.split('m',expand=True)[1]

    data = data.fillna(value='0')
    
    data['writer0']=data['writer0'].astype(np.int32) 
    data['writer1']=data['writer1'].astype(np.int32) 
    data['writer2']=data['writer2'].astype(np.int32) 
    data['writer3']=data['writer3'].astype(np.int32) 
   
    
#    data = data[data['rating']>2]

   
    data.to_csv('data.csv',index=None)
     

    
def renumber_items():
     
    data = pd.read_csv('data.csv')


    items=data['itemID'].reset_index(drop=True)  
    items.drop_duplicates( keep='first', inplace=True)    
    items=items.reset_index(drop=True)
    items=items.reset_index()
    print(' items is ',items)
    data = pd.merge(data, items, how='inner', on=['itemID'])    

    data.rename(columns={'index':'re_itemid'},inplace = True)    
    data.to_csv('data1.csv',columns=[ 'userID', 'rating', 're_itemid','director', 'gerne'],index=0)    




def renumber_users():
    
    data = pd.read_csv('data1.csv')
#    print(data)
    users=data['userID'].reset_index(drop=True)  
    users.drop_duplicates( keep='first', inplace=True)    
    users=users.reset_index(drop=True)
    users=users.reset_index()
    print(users)
    data = pd.merge(data,users, how='inner', on=['userID'])    
#    print(data)
    data.rename(columns={'index':'re_userid'},inplace = True)    
    data.to_csv('data2.csv',columns=[ 're_userid', 'rating', 're_itemid','director', 'gerne'],index=0)    


    
def spilt_train_test():
    data=pd.read_csv('data3.csv')
    data.rename(columns={'re_userid':'userId', 're_itemid':'itemId','re_director':'director'},inplace = True)    
    
    
    posi_data = data[data['rating']>3]
    neg_data =  data[data['rating']<4]
    
    
    test_id = random.sample(range(2536),507)
    train_id = set(list(range(2536)))-set(test_id)    
    test_id = pd.DataFrame(test_id)    
    test_id.rename(columns={0:'itemId'},inplace = True)   
    test_data = pd.merge(posi_data,test_id, how='inner', on=['itemId'])  

    
    train_id = pd.DataFrame(list(train_id))    
    train_id.rename(columns={0:'itemId'},inplace = True) 
    train_data = pd.merge(posi_data,train_id, how='inner', on=['itemId'])  
    

    nega_data = pd.merge(neg_data,train_id, how='inner', on=['itemId'])  
     
    test_data.to_csv('test_data.csv',index=0)
    train_data.to_csv('train_data.csv',index=0)
    nega_data.to_csv('neg_data.csv',index=0)


def renumber_director():  
    
    
    data = pd.read_csv('data2.csv')

    users=data['director'].reset_index(drop=True)  
    users.drop_duplicates( keep='first', inplace=True)    
    users=users.reset_index(drop=True)
    users=users.reset_index()

    data = pd.merge(data,users, how='inner', on=['director'])    

    data.rename(columns={'index':'re_director'},inplace = True)    
    data.to_csv('data3.csv',columns=[ 're_userid', 'rating', 're_itemid','re_director', 'gerne'],index=0)    

    
    

def control():
    
#    get_data()
#    
###    print('renumber_items')
#    renumber_items()
##
###    print('renumber_users')
#    renumber_users()    
#    renumber_director()    

#    print('spilt_train_test')
    spilt_train_test()

    
    
control()
##
##    
a = pd.read_csv('train_data.csv')
print(a)

b = pd.read_csv('test_data.csv')
print(b)


c = pd.read_csv('neg_data.csv')
print(c)


