# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
import pandas as pd 
import numpy as np 
'''  use interaction and item attributes to build user presentation   '''
def user_attribute():
    data = pd.read_csv(r'../data/train_data.csv',usecols=['userId','gerne'])
#    print(data)

    data['tmp']=data['gerne'].str.split('[',expand=True)[1]
    data['tmp1']=data['tmp'].str.split(']',expand=True)[0]

#    print(data)
    user = np.array(data['userId'])
    attr = np.array(data['tmp1'])
     
    print(len(user))
    print(len(attr))
#
#    print(attr[109800])
#    attr_list=np.int32(attr[109800].split(','))
#    print(type(attr_list[1]))
    user_present = np.zeros(shape=(6040,18), dtype= np.int32)
    
    for i in range(len( user )) :
        attr_list=np.int32(attr[i].split(','))   
        for j in  attr_list:
            
            user_present[user[i]][j] += 1.0
      

    save = pd.DataFrame(user_present ) 
    
        
    save =pd.DataFrame(save)
    print(save)  
    save.to_csv('user_attribute.csv',index=0,header=None,columns=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17])
    
user_attribute()
#