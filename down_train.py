#coding:utf-8

import tensorflow as tf
import numpy as np
import load_data
import time
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d
import math
import copy
import keras
from keras import backend as K

class model(object):

    def __init__(self):
        self.model_path='./model/'
        self.var_cnt=0
        self.batch=5
        self.std=0.01
        self.total_len=52608
        self.train_len=43824
        self.val_len=8784
        self.var_list=[]
        self.clip_bounds=[-3.0,3.0]
        self.lamda=0.5
        self.l2=1e-2
        self.change_threshold=2000

    def fcn(self,x,nodes):
        x_shape=x.get_shape()
        self.var_cnt+=1
        w=tf.get_variable('fcn'+str(self.var_cnt),[x_shape[1],nodes],initializer=tf.random_normal_initializer(mean=0,stddev=self.std))
        self.var_cnt+=1
        b=tf.get_variable('fcn'+str(self.var_cnt),[x_shape[0],nodes],initializer=tf.constant_initializer(0.0))
        y=tf.matmul(x,w)+b
        self.var_list.append(w)
        return y

    def unshared_conv(self,x,output_channels,kernel_size=3,strides=1):
        y=keras.layers.LocallyConnected1D(filters=output_channels,
                kernel_size=kernel_size,
                strides=strides,
                activation='relu',
                kernel_initializer='random_uniform',
                bias_initializer='zeros',
                use_bias=True)(x)
        return tf.pad(y,[[0,0],[1,1],[0,0]],mode='CONSTANT')

    def dense_conv(self,x,output_channels):
        shape=x.get_shape()
        if output_channels!=shape[2]:
            x=self.unshared_conv(x,output_channels)
        
        dense1=self.unshared_conv(x,output_channels)+x
        dense1=self.BN(dense1)
        dense1=tf.nn.relu(dense1)

        dense2=self.unshared_conv(dense1,output_channels)+dense1+x
        dense2=self.BN(dense2)
        dense2=tf.nn.relu(dense2)
        return dense2

    def BN(self,x):
        shape=x.get_shape()
        channels=shape[-1]
        mean,variance=tf.nn.moments(x,1,keep_dims=True)
        self.var_cnt+=1
        offset=tf.get_variable('BN'+str(self.var_cnt),shape,initializer=tf.constant_initializer(0.0))   
        self.var_cnt+=1
        scale=tf.get_variable('BN'+str(self.var_cnt),shape,initializer=tf.constant_initializer(1.0))

        return tf.nn.batch_normalization(x,mean,variance,offset,scale,1e-3)

    def build_model(self,x):
        keep_prob=tf.placeholder(tf.float32)

        conv1=self.dense_conv(x,8)
        conv2=self.dense_conv(conv1,16)
        conv3=self.dense_conv(conv2,32)
        conv4=self.dense_conv(conv3,64)
        conv4=tf.nn.dropout(conv4,keep_prob)
        conv5=self.dense_conv(conv4,1)
        shape=conv5.get_shape()
        conv5=tf.reshape(conv5,(shape[0],shape[1]*shape[2]))
        fc1=self.fcn(conv5,10)
        y=self.fcn(fc1,1)
        return y,keep_prob

    def smooth_loss(self,y,y_):
        ter=tf.abs(y-y_)

        smooth_sign=tf.stop_gradient(tf.to_float(tf.less(ter,0.5)))
        loss=tf.pow(ter,2)*0.5*smooth_sign+(ter-0.5)*(1-smooth_sign)

        for var in self.var_list:
            var_sign=tf.stop_gradient(tf.to_float(tf.less(self.clip_bounds[1],var))) + tf.stop_gradient(tf.to_float(tf.less(var,self.clip_bounds[0])))
            tf.add_to_collection('regu',tf.contrib.layers.l2_regularizer(self.l2)(var_sign*var))
        return tf.reduce_mean(loss)+tf.add_n(tf.get_collection('regu'))

    def pro_loss(self,y,y_,sita):
        ter=self.smooth_loss(y,y_)
        sign=tf.stop_gradient(tf.to_float(tf.less(ter,0))) 

        loss=ter*(1-sita)*sign+ter*sita*(1-sign)

        return tf.reduce_mean(loss)

    def loss_L1(self,y,y_):
        return tf.reduce_mean(tf.abs(y-y_))

    def loss_L2(self,y,y_):
        return tf.reduce_mean(tf.square(y-y_))

    def MAPE(self,y,y_):
        ter=tf.abs((y-y_)/y_)
        return tf.reduce_mean(ter)


    def optimizer(self,loss):
        with tf.name_scope('lr'):
            lr=tf.placeholder(tf.float32)
        optimizer=tf.train.AdamOptimizer(lr)
        train_op=optimizer.minimize(loss)
        return train_op,lr

    def norm_compute(self):
        data_loader=load_data.load_data()
        data_list=[]
        for load,data,normal in data_loader.train_fusion_loader(batch_len=1):
            data_list.append(data[0])
        data_list=np.array(data_list)
        data_mean=np.mean(data_list,axis=0)
        data_std=np.std(data_list,axis=0)
        return data_mean,data_std

    def moving_average(self,a,alpha=1):
        alpha_side=(1-alpha)/2
        for i in range(len(a)-1,0,-1):
            if a[i]-a[i-1]>self.change_threshold:
                a[i]=a[i-1]+self.change_threshold
            elif a[i]-a[i-1]<-self.change_threshold:
                a[i]=a[i-1]-self.change_threshold
        ter=[]
        ter.append(a[0])
        for i in range(1,len(a)-1):
            ter.append(alpha_side*a[i-1]+alpha*a[i]+alpha_side*a[i+1])
        ter.append(a[-1])
        return ter

    def train(self,epoches=50):
        self.var_cnt=0
        data_mean,data_std=self.norm_compute()
        with tf.Session() as sess:
            K.set_session(sess)
            x=tf.placeholder(tf.float32,[self.batch,8,1])
            y_=tf.placeholder(tf.float32,[self.batch,1])
            y,keep_prob=self.build_model(x)
            loss_ter=self.pro_loss(y,y_,0.9)
            train_op,lr=self.optimizer(loss_ter)
            saver=tf.train.Saver()
            data_loader=load_data.load_data()
            loss_value=np.zeros((self.val_len/self.batch,),dtype=np.float32)

            sess.run(tf.global_variables_initializer())
            start_time=time.time()
            print 'start training'
            for epoch in xrange(epoches):
                for load,data,normal in data_loader.train_fusion_loader(batch_len=self.batch):
                    data=(data-data_mean)/data_std
                    
                    load_mean,load_std=np.split(normal,2,axis=1)
                    load=(load-load_mean)/load_std
                    
                    feed_dict={x:data,y_:load,lr:1e-4,keep_prob:0.5}
                    op_buffer,y_ter=sess.run([train_op,y],feed_dict=feed_dict)
                     
                cnt=0
                for load,data,normal in data_loader.val_fusion_loader(batch_len=self.batch,val_len=self.val_len):
                    data=(data-data_mean)/data_std
                    
                    load_mean,load_std=np.split(normal,2,axis=1)
                    load=(load-load_mean)/load_std

                    feed_dict={x:data,y_:load,lr:0,keep_prob:1}
                    loss_value[cnt]=sess.run(loss_ter,feed_dict=feed_dict)
                    cnt+=1
                    
                now_time=time.time()
                print('epoch:%d time:%.3f loss:%.5f' %(epoch+1,now_time-start_time,np.mean(loss_value)))
            saver.save(sess,self.model_path+'DCN_down')

if __name__=='__main__':
    a=model()
    a.train()


