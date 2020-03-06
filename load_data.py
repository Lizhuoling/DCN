#coding:utf-8

import csv
from datetime import datetime
import numpy as np

class load_data(object):
    def __init__(self):
        self.Load_List='./data/LoadData.csv'
        self.Weather_List='./data/weather_data_filtered.csv'
        self.total_len=52608
        self.train_len=43824
        self.val_len=8784

    def Load_List_reader(self):
        f=open(self.Load_List,'r')
        reader=csv.reader(f)
        return f,reader

    def Weather_List_reader(self):
        f=open(self.Weather_List,'r')
        reader=csv.reader(f)
        return f,reader

    def train_fusion_loader(self,batch_len):
        Load_f,load_reader=self.Load_List_reader()
        Weather_f,weather_reader=self.Weather_List_reader()

        load_reader.next()
        weather_reader.next()

        for time in range(self.train_len/batch_len):
            Data=[]
            Load=[]
            Normal=[]
            for cnt in range(batch_len):
                load_buff=load_reader.next()
                ter=float(load_buff[1])
                Load.append([ter])
                mean,std=load_buff[3:5]
                mean=float(mean)
                std=float(std)
                Normal.append([mean,std])
                weekday=float(load_buff[2])
        
                weather_buff=weather_reader.next()
                time_buff=weather_buff.pop(0)
                weather_buff=[float(elet) for elet in weather_buff]
                month=float(time_buff[5:7])
                day=float(time_buff[8:10])
                hour=float(time_buff[11:13])
                Data.append(weather_buff+[month,day,hour,weekday])
                
            Data=np.array(Data)
            Data=Data.reshape(Data.shape[0],Data.shape[1],1)
            
            yield np.array(Load),Data,np.array(Normal)

        Load_f.close()
        Weather_f.close()

    def val_fusion_loader(self,batch_len,val_len=8784,skip_len=0,is_vali=True):
        Load_f,load_reader=self.Load_List_reader()
        Weather_f,weather_reader=self.Weather_List_reader()

        for i in range(skip_len+1):
            load_reader.next()
            weather_reader.next()

        if is_vali is True:
            for i in range(self.train_len):
                load_reader.next()
                weather_reader.next()

        for time in range(val_len/batch_len):
            Data=[]
            Load=[]
            Normal=[]
            for cnt in range(batch_len):
                load_buff=load_reader.next()
                ter=float(load_buff[1])
                Load.append([ter])
                mean,std=load_buff[3:5]
                mean=float(mean)
                std=float(std)
                Normal.append([mean,std])
                weekday=float(load_buff[2])
        
                weather_buff=weather_reader.next()
                time_buff=weather_buff.pop(0)
                weather_buff=[float(elet) for elet in weather_buff]
                month=float(time_buff[5:7])
                day=float(time_buff[8:10])
                hour=float(time_buff[11:13])
                Data.append(weather_buff+[month,day,hour,weekday])
                
            Data=np.array(Data)
            Data=Data.reshape(Data.shape[0],Data.shape[1],1)
            
            yield np.array(Load),np.array(Data),np.array(Normal)

        Load_f.close()
        Weather_f.close()

if __name__=='__main__':
    a=load_data()
    for load,data,normal in a.train_fusion_loader(7*24):
        mean,std=np.split(normal,2,axis=1)
