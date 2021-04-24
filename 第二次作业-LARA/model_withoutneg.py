# -*- coding: utf-8 -*-
import pandas as pd 
import tensorflow as tf
import numpy as np
import support
#import matplotlib.pyplot as plt

'''Hyper parameters'''
alpha = 0
attr_num=18 # the number of attribute  
attr_present_dim = 5 #the dimention of attribute present
batch_size = 1024
hidden_dim = 100  # G hidden layer dimention
user_emb_dim = attr_num 
std=0.5


'''D variables'''
D_attri_matrix = tf.get_variable('D_attri_matrix', [2*attr_num, attr_present_dim], initializer=tf.contrib.layers.xavier_initializer())
D_W1 = tf.get_variable('D_w1', [attr_num*attr_present_dim + user_emb_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.get_variable('D_b1', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
D_W2 = tf.get_variable('D_w2', [hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.get_variable('D_b2', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
D_W3 = tf.get_variable('D_w3', [hidden_dim, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())
D_b3 = tf.get_variable('D_b3', [1, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())

D_params = [D_attri_matrix, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

'''G variables'''
G_attri_matrix = tf.get_variable('G_attri_matrix', [2*attr_num, attr_present_dim], initializer=tf.contrib.layers.xavier_initializer())
G_W1 = tf.get_variable('G_w1', [attr_num*attr_present_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.get_variable('G_b1', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
G_W2 = tf.get_variable('G_w2', [hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.get_variable('G_b2', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
G_W3 = tf.get_variable('G_w3', [hidden_dim, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())
G_b3 = tf.get_variable('G_b3', [1, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())

G_params = [G_attri_matrix, G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]


'''placeholder'''

attribute_id  = tf.placeholder(shape=[None, attr_num], dtype=tf.int32)
real_user_emb = tf.placeholder(shape=[None, user_emb_dim], dtype=tf.float32)

neg_attribute_id  = tf.placeholder(shape=[None, attr_num], dtype=tf.int32)
neg_user_emb = tf.placeholder(shape=[None, user_emb_dim], dtype=tf.float32)

'''G'''
def generator(attribute_id):
    attri_present = tf.nn.embedding_lookup(G_attri_matrix, attribute_id) # batch_size x 18 x attr_present_dim
    
    attri_feature = tf.reshape(attri_present, shape=[-1, attr_num*attr_present_dim])
    
    l1_outputs = tf.nn.tanh(tf.matmul(attri_feature, G_W1) + G_b1)
    l2_outputs = tf.nn.tanh(tf.matmul(l1_outputs, G_W2) + G_b2)
    fake_user = tf.nn.tanh(tf.matmul(l2_outputs, G_W3) + G_b3)
    
    return fake_user


def discriminator(attribute_id, user_emb):
    attri_present = tf.nn.embedding_lookup(D_attri_matrix, attribute_id) # batch_size x 18 x attr_present_dim
    attri_feature = tf.reshape(attri_present, shape=[-1, attr_num*attr_present_dim])
    emb = tf.concat([attri_feature, user_emb], 1)
    
    l1_outputs = tf.nn.tanh(tf.matmul(emb, D_W1) + D_b1)
    l2_outputs = tf.nn.tanh(tf.matmul(l1_outputs, D_W2) + D_b2)
    D_logit = tf.matmul(l2_outputs, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit

'''loss'''
fake_user_emb = generator(attribute_id)
D_real, D_logit_real = discriminator(attribute_id, real_user_emb)
D_fake, D_logit_fake = discriminator(attribute_id, fake_user_emb)

D_counter, D_logit_counter = discriminator(neg_attribute_id, neg_user_emb)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))

D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss_counter = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_counter, labels=tf.zeros_like(D_logit_counter))) 

D_regular = alpha * (tf.nn.l2_loss(D_attri_matrix) + tf.nn.l2_loss(D_W1) + tf.nn.l2_loss(D_b1) + tf.nn.l2_loss(D_W2) + tf.nn.l2_loss(D_b2) + tf.nn.l2_loss(D_W3) + tf.nn.l2_loss(D_b3))
G_regular = alpha * (tf.nn.l2_loss(G_attri_matrix) + tf.nn.l2_loss(G_W1) +
                     tf.nn.l2_loss(G_b1) + tf.nn.l2_loss(G_W2) + tf.nn.l2_loss(G_b2) + tf.nn.l2_loss(G_W2) + tf.nn.l2_loss(G_b2)+tf.nn.l2_loss(G_W3)+ tf.nn.l2_loss(G_b3) )

D_loss = (1-alpha)*(D_loss_real + D_loss_fake) + D_regular
G_loss = (1-alpha)*(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))) + G_regular

'''optimizer'''
D_solver = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(D_loss, var_list=D_params)
G_solver = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(G_loss, var_list=G_params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())


test_item_batch,  test_attribute_vec=support.get_testdata()

test_G_user = sess.run(fake_user_emb, feed_dict={attribute_id: test_attribute_vec})

p_at_10, p_at_20, M_at_10, M_at_20, G_at_10, G_at_20 = support.test(test_item_batch, test_G_user)
print('at start p_at_10 is', p_at_10, 'p_at_20', p_at_20, 'M_at_10', M_at_10, 'M_at_20', M_at_20,'G_at_10',G_at_10,'G_at_20',G_at_20)


#max_p_at_k = 0
max_p_at_10 = 0
max_p_at_20 = 0
max_M_at_10 = 0
max_M_at_20 = 0 
max_G_at_10 = 0
max_G_at_20 = 0

train_data = np.array(pd.read_csv(r'data/train_data.csv')) 

np.random.shuffle(train_data)
user_emb_matrix = np.array(pd.read_csv(r'util/user_emb.csv', header=None))
 
 
p_10_to_save = []
p_20_to_save = []
M_10_to_save = []
M_20_to_save = []
G_10_to_save = []
G_20_to_save = []
 
'''train'''
for it in range(400):
    D_range = 1
    G_range = 1    
    
    support.shuffle()
    support.shuffle2()
    for D_it in range(D_range):
        print('ddd')
        index = 0
        while index < 253236:
             if index + batch_size <= 253236:
                 train_user_batch, train_item_batch, train_attr_batch,train_user_emb_batch = support.get_traindata(index, index + batch_size)
                 counter_user_batch, counter_item_batch, counter_attr_batch, counter_user_emb_batch = support.get_negdata(index, index + batch_size)
             index = index + batch_size
               
             _, D_loss_now, fake_us = sess.run([D_solver, D_loss, fake_user_emb],
                                      feed_dict={attribute_id: train_attr_batch,
                                                 real_user_emb: train_user_emb_batch,
                                                 neg_attribute_id:  counter_attr_batch,
                                                 neg_user_emb: counter_user_emb_batch
                                               })  
        print(D_loss_now)

    for G_it in range(G_range):
        index = 0
        while index < 253236:
             if index + batch_size <= 253236:
                 train_user_batch, train_item_batch, train_attr_batch,train_user_emb_batch= support.get_traindata(index, index + batch_size)
             index = index + batch_size
             
             _, G_loss_now = sess.run([G_solver, G_loss], feed_dict={attribute_id:train_attr_batch})
        print(G_loss_now)
    if it % 1 == 0:
        test_item_batch,  test_attribute_vec= support.get_testdata()
        test_G_user = sess.run(fake_user_emb, feed_dict={attribute_id: test_attribute_vec})
#        print( test_G_user[:10])
        p_at_10, p_at_20, M_at_10, M_at_20, G_at_10, G_at_20 = support.test(test_item_batch, test_G_user)
        if p_at_10 > max_p_at_10:
            max_p_at_10 = p_at_10
        p_10_to_save.append(p_at_10)
        if p_at_20 > max_p_at_20:
            max_p_at_20 = p_at_20
        p_20_to_save.append(p_at_20)
        if M_at_10 > max_M_at_10:
            max_M_at_10 = M_at_10
        M_10_to_save.append(M_at_10)
        if M_at_20 > max_M_at_20:
            max_M_at_20 = M_at_20
        M_20_to_save.append(M_at_20)
        if G_at_10 > max_G_at_10:
            max_G_at_10 = G_at_10
        G_10_to_save.append(G_at_10)
        if G_at_20 > max_G_at_20:
            max_G_at_20 = G_at_20
        G_20_to_save.append(G_at_20)
      
        print('it', it, 'p_at_10 ', p_at_10, 'p_at_20', p_at_20, 'M_at_10', M_at_10, 'M_at_20', M_at_20, 'G_at_10', G_at_10, 'G_at_20', G_at_20)
        print('max p_at_10 ', max_p_at_10, 'p_at_20', max_p_at_20, 'M_at_10', max_M_at_10, 'M_at_20', max_M_at_20, 'G_at_10', max_G_at_10, 'G_at_20', max_G_at_20)
    if it % 100 == 0:
        pd.DataFrame(p_10_to_save).to_csv('p10.csv')
        pd.DataFrame(p_20_to_save).to_csv('p20.csv')
        pd.DataFrame(M_10_to_save).to_csv('m10.csv')
        pd.DataFrame(M_20_to_save).to_csv('m20.csv')
        pd.DataFrame(G_10_to_save).to_csv('g10.csv')
        pd.DataFrame(G_20_to_save).to_csv('g20.csv')
      
        











































