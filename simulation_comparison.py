# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 10:54:32 2023

@author: 77359
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn import tree
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import sklearn.svm as svm

from collections import Counter
import math
import time
import copy
import random
import operator
# from scipy.spatial import distance
import functools as fc
np.seterr(all='ignore',divide='ignore',invalid='ignore')
np.random.seed(19) 

from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_hastie_10_2
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

# 获取变量名称
def namestr(obj, namespace):
    l=[name for name in namespace if namespace[name] is obj]
    s=str(l)[2:-2]
    return s

a=[1,2,3]
b=namestr(a,globals())
b

# 0. Main program
# 0.1 Main function_comparison
# Input：
# DataSet: data array
# method:mass function(num、Prob available），
# t:Interval number，
# h:Boundary threshold，
# fA:The proportion of undersampling,
# fB:The proportion of SMOTE, 
# n:Number of cycles;
# M:The folds of cross-validation
# Output：classification results

def main(DataSet, method, t, h1, hz, fA, fB, n, M, sample_method):
#     DataSet=loadDataSet(path)
    X=DataSet[:,:-1]
    y=DataSet[:,-1]
    N=X.shape[0]
    d=X.shape[1]
    
    for i in range(n):
        T=[]
        IMB=[]
        P=[]
        R=[]
        F1=[]
        G=[]
        Ks=[]
        Auc=[]
        Ari=[]

        start = time.perf_counter()
        if sample_method == 'RS':   #Random sampling
            D=us(DataSet,fA)
        elif sample_method == 'RUS':   #Random undersampling
            D=rus(DataSet)
        elif sample_method == 'SMOTE':  #SMOTE
            D=smote(DataSet,fB)
        elif sample_method == 'NM':  #Nearmiss
            D=nm1(DataSet)
        else: #NGBM
            #Data points are mapped to grid sequences
            MM,Delta=mm(X,t)
            Cell,ACell=cell(X,Delta,t)
            #Get grid sequence information
            I=num_count(ACell,y,t,d)
            Bound=bound(I,t,d,method)
            #Grid marker
            CC=compare_h(Bound,h1,hz,d)   #Grid sequence marker
            DM=datamark(ACell,CC)   #Dataset marker
            #Mixed sampling
            D=sample(DataSet,DM,fB)
        
        end = time.perf_counter()
        tt = end - start
        imb=len(D[np.where(D[:,-1]==0)])/len(D[np.where(D[:,-1]==1)])    
        #classification learning
        p,r,f1_score,g_mean,ks,auc,ari=re_CLF(D,M)
        
        T.append(tt)
        IMB.append(imb)
        P.append(p)
        R.append(r)
        F1.append(f1_score)
        G.append(g_mean)
        Ks.append(ks)
        Auc.append(auc)
        Ari.append(ari)
        
    ave_t=np.nanmean(T)
    ave_imb=np.nanmean(IMB)
    ave_p=np.nanmean(P)
    ave_r=np.nanmean(R)
    ave_f1score=np.nanmean(F1)
    ave_gmean=np.nanmean(G)
    ave_ks=np.nanmean(Ks)
    ave_auc=np.nanmean(Auc)
    ave_ari=np.nanmean(Ari)
            
    return ave_t,ave_imb,ave_p,ave_r,ave_f1score,ave_gmean,ave_ks,ave_auc,ave_ari
        

# 1. Data gridding
# 1.1 Extract the maximum and minimum values for each dimension
#Gets the maximum and minimum values of each column -- the upper and lower limits of the D-dimensional bounded space of X，
#Input：dataset X
#Ouptput：A D-dimensional list of extreme values for each column of the dataset
def max_v(martix):
    res_list=[]
    for j in range(len(martix[1])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append(int(martix[i][j])+1)
        res_list.append(max(one_list))
    return res_list

def min_v(martix):
    res_list=[]
    for j in range(len(martix[1])):
        one_list=[]
        for i in range(len(martix)):
            one_list.append(int(martix[i][j]))
        res_list.append(min(one_list))
    return res_list

# Input: dataset: DataSet（N, ind+d+1），Interval number: t
# Output：Extremal matrix: MM（2, d），Step size of each dimension of grid space: Delta（d,)
def mm(X, t):
    Max=max_v(X)
    Min=min_v(X)
    MM=np.vstack((np.array(Max),np.array(Min)))
    
    Dis=MM[0,:]-MM[1,:]
    Delta=Dis/t
    
    return MM,Delta

# 1.2 Grid mapping: Returns the grid index matrix
# Input: dataset: X（N, d），Step matrix of grid space: Delta（d,）
# Output：Grid index matrix:Cell (N, d)，Grid sequence index matrix:ACell list（N,1）
def cell(X,Delta,t):
    Min=min_v(X)
    P=[]
    for xi in X:
        pi=xi-np.array(Min)
        P.extend(pi.tolist())
    P=np.array(P).reshape((X.shape[0],X.shape[1]))
    ind=[di/Delta for di in P]
    Cell=np.trunc(ind)
    Cell=Cell.astype(int)
    
    ACell=[]
    d=X.shape[1]
    for ci in Cell:
        ai=0
        for i in range(d):
            ai=ai+ci[i]*(t**(d-i-1))
        ACell.append(ai)
    
    return Cell,ACell

# 2. Boundary domain identification
# 2.1 Boundary degree reference index: the number of positive/negative class samples in each grid
# Input：Grid sequence index list:ACell(N，)，datset:y(N,1), Interval number: t, dimensions: d
# Output：Grid index positive and negative number of class samples: list(t**d, 2)
def num_count(ACell,y,t,d):
    I_num=np.zeros(t**d*2).reshape(t**d,2)
    for a in ACell:
        I_num[a,1]+=y[ACell.index(a)]
        I_num[a,0]+=abs(y[ACell.index(a)]-1)

    return I_num

# 2.2 Boundary degree calculation
# (2) Calculate boundary degree|k{q}-k{p}|/k{p&q}*n
# Input：Grid sequence information list:I，Calculated boundary degree
# Output：Grid boundary degree: Bound（2, t**d, d）

def bound(I_num,t,d,method):
    B=np.ones(2*(t**d)*d).reshape((2,t**d,d))
    if method == 'num':
        I=I_num[:,1]  
        for io in range(len(I)):
            ii=I[io]
            for di in range(d):
    #             print('现在是第【{0}】个网格，维数是{1}'.format(io,di))
                if io-t**(d-di-1)>=0:
                    if I[io]+I[io-t**(d-di-1)] != 0:
                        B[0,io,di]=abs(I[io]-I[io-t**(d-di-1)])/(I[io]+I[io-t**(d-di-1)])     
                    else:
                        B[0,io,di]=abs(I[io]-I[io-t**(d-di-1)])/(1+I[io]+I[io-t**(d-di-1)])     
                else:
                    B[0,io,di]=1
                if io+t**(d-di-1)<len(I):
                    if I[io]+I[io-t**(d-di-1)] != 0:
    #                 print('右邻居索引：',io+t**(d-di-1))
                        B[1,io,di]=abs(I[io]-I[io+t**(d-di-1)])/(I[io]+I[io+t**(d-di-1)])
                    else:
                        B[1,io,di]=abs(I[io]-I[io+t**(d-di-1)])/(1+I[io]+I[io+t**(d-di-1)])
                else:            
                    B[1,io,di]=1
    else:
        N_num=np.zeros(2*(t**d)).reshape((t**d,2))
        I=I_num
        for io in range(len(I)):
            for di in range(d):
                #每一维保存该单元格邻居的负类数和正类数，0,2,4保存负类，1,3,5保存正类数
                if io-t**(d-di-1)>=0 and io+t**(d-di-1)<len(I):
                    N_num[io,0]+=(I[io-t**(d-di-1),0]+I[io+t**(d-di-1),0])
                    N_num[io,1]+=(I[io-t**(d-di-1),1]+I[io+t**(d-di-1),1])
                    
        for io in range(len(I)):
            for di in range(d):
    #             print('现在是第【{0}】个网格，维数是{1}'.format(io,di))
                if io-t**(d-di-1)>=0:
                    B[0,io,di]=abs((N_num[io,1]/N_num[io,0])-(N_num[io-t**(d-di-1),1]/N_num[io-t**(d-di-1),0])) / (N_num[io,1]+N_num[io-t**(d-di-1),1]-I[io,1]-I[io-t**(d-di-1),1]) * (N_num[io,0]+N_num[io-t**(d-di-1),0]-I[io,0]-I[io-t**(d-di-1),0])       
                else:
                    B[0,io,di]=1
                if io+t**(d-di-1)<len(I):
    #                 print('右邻居索引：',io+t**(d-di-1))
                    B[1,io,di]=abs((N_num[io,1]/N_num[io,0])-(N_num[io+t**(d-di-1),1]/N_num[io+t**(d-di-1),0])) / (N_num[io,1]+N_num[io+t**(d-di-1),1]-I[io,1]-I[io+t**(d-di-1),1]) * (N_num[io,0]+N_num[io+t**(d-di-1),0]-I[io,0]-I[io+t**(d-di-1),0])
                else:            
                    B[1,io,di]=1
                    
    return B

# 2.3 Threshold comparison
# Input：Grid boundary degree:Bound（2，t^d, d），h=0.32
# Output：Threshold comparison array:（2，t^d, d）
def compare_h(Bound,h1,hz,d):
    a,b,c=Bound.shape
    #边界阈值h1
    H1=h1*np.ones(a*b*c).reshape((a,b,c))
    CH1=Bound>=H1
    CL1=CH1[0,:,:]
    V1=pd.DataFrame(CL1).fillna(0)
    CL1=np.array(V1)
    CR1=CH1[1,:,:]
    V1=pd.DataFrame(CR1).fillna(0)
    CR1=np.array(V1)
    C1=CL1*1+CR1*1
    CB1=np.sum(C1,axis=1)
    CC1=CB1>0
    CC1=CC1*1
    
    for i in range(len(CB1)):
        if CB1[i]>=c*2:
            CC1[i]=-1 
            
    #边界阈值h1
    Hz=hz*np.ones(a*b*c).reshape((a,b,c))
    CHz=Bound>=Hz
    CLz=CH1[0,:,:]
    Vz=pd.DataFrame(CLz).fillna(0)
    CLz=np.array(Vz)
    CRz=CHz[1,:,:]
    Vz=pd.DataFrame(CRz).fillna(0)
    CRz=np.array(Vz)
    Cz=CLz*1+CRz*1
    CBz=np.sum(Cz,axis=1)
    CCz=CBz>0
    CCz=CCz*1
    
    for i in range(len(CBz)):
        if CBz[i]>=c*2:
            CCz[i]=-1
    
    CCz[np.where(CCz[:]==1)]=0
    
    CC=CC1+CCz
                
    return CC

# 2.4 Data set boundary marker
# Input：Grid sequence index matrix:ACell list (N,1)，Grid sequence marker:CC list(t**d，A0|B1|Z-1)
# Output：Data set marker:DataMark list (N,A0|B1|Z-1)

def datamark(ACell,CC):
    DataMark=[]
    for n in ACell:
        DataMark.append(CC[n])
    return DataMark

# 3. sampling methods
# 3.0 Random undersampling

def rus(DataSet):      
    Xu=DataSet[:,:-1]
    yu=DataSet[:,-1]
    yu=yu.reshape((len(yu),1))
    rus = RandomUnderSampler(random_state=0)
    X_rus, y_rus = rus.fit_resample(Xu, yu)
    DA=np.hstack((X_rus,y_rus.reshape(len(y_rus),1)))
    
    return DA

# 3.1 Random sampling

def us(DataSetA0,fA):    
    Data=pd.DataFrame(DataSetA0)
    weights=np.ones(len(Data))
    D=pd.DataFrame.sample(Data,None,fA,weights.any())
    DA0=np.array(D)
    
    return DA0

# 3.2 SMOTE

def smote(DataSetB1,fB):
    smo=SMOTE(random_state=19)
    Xk=DataSetB1[:,:-1]
    yk=DataSetB1[:,-1]
    yk=yk.reshape((len(yk),1))
    X_smo,y_smo=smo.fit_resample(Xk,yk)
    DB=np.hstack((X_smo,y_smo.reshape(len(y_smo),1)))

    return DB

# 3.3 Nearmiss
from imblearn.under_sampling import NearMiss

def nm1(DataSet):      
    Xu=DataSet[:,:-1]
    yu=DataSet[:,-1]
    yu=yu.reshape((len(yu),1))
    rus = NearMiss(version=1, n_neighbors=3)
    X_rus, y_rus = rus.fit_resample(Xu, yu)
    DC=np.hstack((X_rus,y_rus.reshape(len(y_rus),1)))
    
    return DC

# 3.4 NGBM

def sample(DataSet,DataMark,fB):
    DataMark=np.array(DataMark).reshape(len(DataMark),1)
    DataSet0=DataSet[np.where(DataSet[:,-1]==0)]
    DataSet1=DataSet[np.where(DataSet[:,-1]==1)]


    DataSetA=DataSet[np.where(DataMark[:,0]==0)]
    

    DataSetB=DataSet[np.where(DataMark[:,0]==1)]
    DataSetB1=DataSetB[np.where(DataSetB[:,-1]==1)]
    DataSetB0=DataSetB[np.where(DataSetB[:,-1]==0)]
    
    fBB=len(DataSet0)/len(DataSet1)
    
    def os(DataSetB1,fB):
        a,b=np.shape(DataSetB1)
        A=int(a*fB)
        D0=np.zeros(A*b).reshape(A,b)
        DataSetB1=np.vstack((DataSetB1,D0))
       
        smo=SMOTE(random_state=19)
        Xk=DataSetB1[:,:-1]
        yk=DataSetB1[:,-1]
        yk=yk.reshape((len(yk),1))
        X_smo,y_smo=smo.fit_resample(Xk,yk)
        DB=np.hstack((X_smo,y_smo.reshape(len(y_smo),1)))
        DB=DB[np.where(DB[:,-1]==1)]
        
        return DB
    
    DB1=os(DataSetB1,fBB)
    
    Dos=np.vstack((DB1,DataSetB0,DataSetA))
    Ds=rus(Dos)

    return Ds

# 4. classification learning
# 4.1 K-fold cross-validation
# Input：k: folds，dataset: D
# output：Training set index:train_ind，testing set index:test_ind

def k_fold(k,data):
    train_index=[]
    test_index=[]
    kf=KFold(n_splits=k,shuffle=True,random_state=110)
    d_t=kf.split(data)
    for train, test in d_t:
        train_index.append(train.tolist())
        test_index.append(test.tolist())
        
    return train_index,test_index

# 4.2 ROC
# Input：Test set labels:y_test, Predict set:y_predict
# Output: ROC(picture)
def plot_roc(y_test, y_predict):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_predict)
    roc_auc = metrics.auc(false_positive_rate, true_positive_rate) #计算AUC值
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    # plt.savefig('figures/PC5.png')
    plt.show()

# 4.3 classifier
# Decision Tree:DT
def CLF(X_train,y_train,X_test,y_test):
    clf = DecisionTreeClassifier(class_weight=None)
    clf.fit(X_train,y_train)
    y_predict=clf.predict(X_test)
    conf_mat=metrics.confusion_matrix(y_test, y_predict)
    report=metrics.classification_report(y_test, y_predict)
    p = metrics.precision_score(y_test, y_predict, average='binary')
    r = metrics.recall_score(y_test, y_predict, average='binary')
    f1_score = metrics.f1_score(y_test, y_predict, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_predict)
    ks=max(tpr-fpr)
    G=((tpr*(1-fpr))**0.5)
    g_mean=G[1]
    auc=metrics.auc(fpr,tpr)
    plot_roc(y_test, y_predict)
    ari=metrics.adjusted_rand_score(y_test,y_predict)

    return conf_mat, report, p, r, f1_score, g_mean, ks, auc, ari

# bagging:bag
from sklearn.ensemble import BaggingClassifier

def CLF(X_train,y_train,X_test,y_test):
    clf = BaggingClassifier(random_state=29)
    clf.fit(X_train,y_train)
    y_predict=clf.predict(X_test)
    conf_mat=metrics.confusion_matrix(y_test, y_predict)
    report=metrics.classification_report(y_test, y_predict)
    p = metrics.precision_score(y_test, y_predict, average='binary')
    r = metrics.recall_score(y_test, y_predict, average='binary')
    f1_score = metrics.f1_score(y_test, y_predict, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_predict)
    ks=max(tpr-fpr)
    G=((tpr*(1-fpr))**0.5)
    g_mean=G[1]
    auc=metrics.auc(fpr,tpr)
    plot_roc(y_test, y_predict)
    ari=metrics.adjusted_rand_score(y_test,y_predict)

    return conf_mat, report, p, r, f1_score, g_mean, ks, auc, ari

# adaboost:ada
from sklearn.ensemble import AdaBoostClassifier

def CLF(X_train,y_train,X_test,y_test):
    clf = AdaBoostClassifier(random_state=29,n_estimators=100)
    clf.fit(X_train,y_train)
    y_predict=clf.predict(X_test)
    conf_mat=metrics.confusion_matrix(y_test, y_predict)
    report=metrics.classification_report(y_test, y_predict)
    p = metrics.precision_score(y_test, y_predict, average='binary')
    r = metrics.recall_score(y_test, y_predict, average='binary')
    f1_score = metrics.f1_score(y_test, y_predict, average='binary')
    fpr, tpr, thresholds = metrics.roc_curve(y_test,y_predict)
    ks=max(tpr-fpr)
    G=((tpr*(1-fpr))**0.5)
    g_mean=G[1]
    auc=metrics.auc(fpr,tpr)
    plot_roc(y_test, y_predict)
    ari=metrics.adjusted_rand_score(y_test,y_predict)

    return conf_mat, report, p, r, f1_score, g_mean, ks, auc, ari

# 4.4 n times learning，Return index mean
# Input：Data set to be learned: D，Cycle number:n
# Output：accuracy，recall，f1_score，g_mean，auc.

def re_CLF(D,M):
    ind=k_fold(M,D)
    ind_train=ind[0]
    ind_test=ind[1]

    Precision=[]
    Recall=[]
    F1_SCORE=[]
    G_MEAN=[]
    KS=[]
    AUC=[]
    ARI=[]

    for i in range(M):
        train_ind=ind_train[i]
        test_ind=ind_test[i]
        
        D_train=D[train_ind]
        X_train=D_train[:,:-1]
        y_train=D_train[:,-1]
        
        D_test=D[test_ind]
        X_test=D_test[:,:-1]
        y_test=D_test[:,-1] 

        RE=CLF(X_train,y_train,X_test,y_test)
        print('The Result of CLF the {0} time'.format(i))

        conf_mat=RE[0]
        report=RE[1]
        p=RE[2]
        r=RE[3]
        f1_score=RE[4]
        g_mean=RE[5]
        ks=RE[6]
        auc=RE[7]
        ari=RE[8]

        Precision.append(p)
        Recall.append(r)
        F1_SCORE.append(f1_score)
        G_MEAN.append(g_mean)
        KS.append(ks)
        AUC.append(auc)
        ARI.append(ari)

    return np.mean(Precision),np.mean(Recall),np.mean(F1_SCORE),np.mean(G_MEAN),np.mean(KS),np.mean(AUC),np.mean(ARI)

# 5. simulation data set
imb=100 # imbalance degree
n1=1000 #the number of the positive samples
n0=n1*imb
Nu=n0+n1
w0=n0/Nu
w1=n1/Nu

# CD
X1,y1= make_blobs(n_samples=[n0,n1], n_features=2,centers=[[-1,-1],[1,1]],cluster_std=[1,0.5],shuffle=True,random_state=29)
y1=y1.reshape(len(y1),1)
DataSet1=np.hstack((X1,y1))
np.shape(DataSet1)

# LD
X2,y2= make_classification(n_samples=Nu,weights=[w0,w1],n_features=2,n_redundant=0,n_informative=1,
                           n_clusters_per_class=1,n_classes=2,random_state=29)
y2=y2.reshape(len(y2),1)
DataSet2=np.hstack((X2,y2))
np.shape(DataSet2)

# HD
X3,y3 = make_moons(n_samples=(n0,n1),shuffle=False, noise=0.2,random_state=29)
y3=y3.reshape(len(y3),1)
DataSet3=np.hstack((X3,y3))
np.shape(DataSet3)

# Data sets, sampling methods cyclic dictionary
DataSet_list=[DataSet1,DataSet2,DataSet3]
Sample_list=['RUS','SMOTE','NM','NGBM']

# 6.0 Experimental setting
t=238
h1=0.75
hz=0.8   
fA=0.8  
fB=1.1   
n=3   
M=5   

writer = pd.ExcelWriter('res.xlsx')

# 6.1 Method=‘num’
method='num'

RE=[]
NUM_excel=pd.DataFrame(index=['time','Imbalanced','precision','recall','F1_score','G_mean','KS','AUC','ARI'])  
for dataset in DataSet_list:
    for samplemethod in Sample_list:
        RE=[]
        print('==================================')
        print('DATA：',namestr(dataset,globals())[:8])
        print('SAMPLE：',samplemethod)
        tt,imb,p,r,f1_score,g_mean,ks,auc,ari=main(dataset, method, t, h1, hz, fA, fB, n, M, samplemethod)
        RE=[tt,imb,p,r,f1_score,g_mean,ks,auc,ari]       
        NUM_excel[namestr(dataset,globals())[:8]+samplemethod]=RE
        print('OVER')
        print('==================================')
NUM_excel.to_excel(writer,sheet_name='NUM-DT')

# 6.2 Method=‘prob’
method='prob'

RE=[]
PROB_excel=pd.DataFrame(index=['time','Imbalanced','precision','recall','F1_score','G_mean','KS','AUC','ARI'])  
for dataset in DataSet_list:
    for samplemethod in Sample_list:
        RE=[]
        print('==================================')
        print('DATA：',namestr(dataset,globals())[:8])
        print('SAMPLE：',samplemethod)
        tt,imb,p,r,f1_score,g_mean,ks,auc,ari=main(dataset, method, t, h1, hz, fA, fB, n, M, samplemethod)
        RE=[tt,imb,p,r,f1_score,g_mean,ks,auc,ari]          
        PROB_excel[namestr(dataset,globals())[:8]+samplemethod]=RE
        print('OVER')
        print('==================================')
PROB_excel.to_excel(writer,sheet_name='PROB-DT')

# save excel
writer._save()
writer.close()

