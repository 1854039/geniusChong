

# 创建文件夹
from flask import json
from scipy import cluster

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import csv
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from torch.autograd import Variable


class FaceLandmarksDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file, iterator=True)

    def __len__(self):

        return 149400

    def __getitem__(self, idx):

        landmarks = self.landmarks_frame.get_chunk(10).values.astype('float')
        #landmarks = self.landmarks_frame.ix[idx, 1:].as_matrix().astype('float')

        # 采用这个，不错。
        return landmarks



def read_data(test_data=r'C:\Users\lin\Desktop\Design\Pro_Data\data_temp.csv', n=0, label=1):
    '''
    加载数据的功能
    n:特征数据起始位
    label：是否是监督样本数据
    '''

    df=pd.read_csv(test_data,encoding="gbk")


   # new=preprocessing.StandardScaler().fit_transform(df.iloc[:,3:16]).round(2)

    print("start")
    # newrr=pd.DataFrame(new)
    # newrr.to_csv(k)
    df['target']=pd.cut(x=df['chl/mg m-3'], bins=[-1.5, 0, 1.5, 3, 4.5,6], labels=[1, 2, 3, 4,5], right=True)
    df.to_csv(test_data)
    csv.field_size_limit(500 * 1024 * 1024)  # 一定要加上这一句
    csv_reader = csv.reader(open(test_data, encoding="utf-8", errors="ignore"))

    data_list = []
    for one_line in csv_reader:
        data_list.append(one_line)
    x_list = []
    y_list = []

    for one_line in data_list[1:]:
        if label == 1:  # 如果是监督样本数据
            y_list.append(int(float(one_line[-1])))  # 标志位(最后一位都是标签位)
            one_list = [o for o in one_line[n:-1]]
            x_list.append(one_list)
        else:
            one_list = [o for o in one_line[n:]]
            x_list.append(one_list)
    return x_list, y_list
def get_distribution_sampler(mu, sigma, batchSize, FeatureNum):
    """
    Generate Target Data, Gaussian
    Input
    - mu: 均值
    - sugma: 方差
    Output
    """
    return Variable(torch.Tensor(np.random.normal(mu, sigma, (batchSize, FeatureNum))))

def split_data(data_list, y_list, ratio=0.30):  # 70%训练集，30%测试集: 914285,391837
    '''
    按照指定的比例，划分样本数据集
    ratio: 测试数据的比率
    '''
    X_train, X_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio, random_state=50)

    """训练集"""
    with open(r'C:\Users\lin\PycharmProjects\Red_tide\Data\train\train.csv', 'w', encoding="utf8", newline="",
              errors="ignore") as csvfile:  # 不加newline=""的话会空一行出来
        fieldnames = ['lat', 'lon', 'depth/m','chl/mg m-3','o2/mmol m-3','no3/mmol m-3','po4/mmol m-3','si/mmol m-3','nppv/mg m-3 day-1','eastward_velocity/m/s','northward_velocity/m/s','wind_speed/m/s','wind_stress/Pa','par/einstein m-2 day-1','sst/℃','zos/m','target']
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        write.writeheader()  # 写表头
        for i in range(len(X_train)):
            write.writerow({'lat': X_train[i][1], 'lon': X_train[i][2], 'depth/m':X_train[i][3],'chl/mg m-3':X_train[i][4],'o2/mmol m-3':X_train[i][5],'no3/mmol m-3':X_train[i][6],'po4/mmol m-3':X_train[i][7],'si/mmol m-3':X_train[i][8],'nppv/mg m-3 day-1':X_train[i][9],'eastward_velocity/m/s':X_train[i][10],'northward_velocity/m/s':X_train[i][11],'wind_speed/m/s':X_train[i][12],'wind_stress/Pa':X_train[i][13],'par/einstein m-2 day-1':X_train[i][14],'sst/℃':X_train[i][15],'zos/m':X_train[i][16],'target':y_train[i]})
    #with open(r'C:\Users\lin\PycharmProjects\Red_tide\Data\train\label.csv', 'w') as fp:
    pd.DataFrame(y_test).to_csv(r'C:\Users\lin\PycharmProjects\Red_tide\Data\train\label.csv', mode='a', header=False, index=False)
    """测试集"""
    # 测试csv
    with open(r'C:\Users\lin\PycharmProjects\Red_tide\Data\test\test.csv', 'w', encoding="utf8", newline="",
              errors="ignore") as csvfile:  # 不加newline=""的话会空一行出来
        fieldnames = ['lat', 'lon',  'depth/m','chl/mg m-3']
        write = csv.DictWriter(csvfile, fieldnames=fieldnames)
        write.writeheader()  # 写表头
        for i in range(len(X_test)):
            write.writerow({'lat': X_train[i][1], 'lon': X_train[i][2], 'depth/m':X_train[i][3],'chl/mg m-3':X_train[i][4]})
    return X_train, X_test,y_train,y_test


if __name__ == '__main__':
    """获取大文件的数据"""
    x_list, y_list = read_data()
    #data1 = pd.read_csv(r'C:\Users\lin\Desktop\Design\Pro_Data\data_temp.csv',encoding="gbk")
    #data2=pd.read_csv(r'C:\Users\lin\PycharmProjects\Red_tide\Data\test.csv')
    # print(data1['chl/mg m-3'].max())
    # print(data1['chl/mg m-3'].min())
   # print(data2['chl/mg m-3'].max())
    """划分为训练集和测试集及label文件"""

    split_data(x_list, y_list)
    print("OK")



def to_var(x):
    """ Make a tensor cuda-erized and requires gradient """
    return to_cuda(x).requires_grad_()

def to_cuda(x):
    """ Cuda-erize a tensor """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def load_data():
    train = pd.read_csv('./Data/train/train.csv',iterator=True)
    test=pd.read_csv('./Data/train/test.csv',iterator=True)
    train_pre = preprocessing.StandardScaler().fit_transform(train)
    test_pre = preprocessing.StandardScaler().fit_transform(test)

    return train_pre,test_pre

def get_data(BATCH_SIZE=100):
    """ Load data for binared MNIST """
    torch.manual_seed(3435)

    # Download our data

    # # Use greyscale values as sampling probabilities to get back to [0,1]
    # train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    # train_label = torch.LongTensor([d[1] for d in train_dataset])
    #
    # test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    # test_label = torch.LongTensor([d[1] for d in test_dataset])
    #
    # # MNIST has no official train dataset so use last 10000 as validation
    # val_img = train_img[-10000:].clone()
    # val_label = train_label[-10000:].clone()
    #
    # train_img = train_img[:-10000]
    # train_label = train_label[:-10000]

    # Create data loaders
    filename1 = r'C:\Users\lin\PycharmProjects\Red_tide\Data\test\test.csv'
    filename = r'C:\Users\lin\PycharmProjects\Red_tide\Data\train\train.csv'
    filename2= r'C:\Users\lin\PycharmProjects\Red_tide\Data\train\label.csv'

    train_dataset = FaceLandmarksDataset(filename)
    test_dataset = FaceLandmarksDataset(filename1)
    label_dataset = FaceLandmarksDataset(filename2)
    # train_dataset=train_dataset[0:1000,:]
    # test_dataset = test_dataset[0:1000,:]
    # torch.from_numpy(d for d in train_dataset)
    # torch.from_numpy(d for d in test_dataset)
    # train_img = torch.stack([torch.bernoulli(d[0]) for d in train_dataset])
    # train_label = torch.LongTensor([d[1] for d in train_dataset])
    #
    # test_img = torch.stack([torch.bernoulli(d[0]) for d in test_dataset])
    # test_label = torch.LongTensor([d[1] for d in test_dataset])
    #
    # # Create data loaders
    # train = torch.utils.data.TensorDataset(train_img, train_label)
    # test = torch.utils.data.TensorDataset(test_img, test_label)
    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)
   # label_iter = torch.utils.data.DataLoader(label_dataset, batch_size=100, shuffle=True)
    print(int(len(train_iter)))
    label_iter=torch.utils.data.DataLoader(label_dataset, batch_size=200, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True)

    # train = torch.utils.data.TensorDataset(train_img, train_label)
    # test = torch.utils.data.TensorDataset(test_img, test_label)
    #
    # train_iter = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    # test_iter = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=True)

    return train_iter,test_iter,test_dataset,train_dataset,label_iter
