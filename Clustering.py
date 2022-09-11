import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from tqdm import tnrange
import scipy.stats as ss 
import math
import warnings
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import DownloadData as dd
import LoadData as ld
import MutualInformation as mf
class Clustering ():
    def __init__ (self, Path, NameStocks, NameSaving, StartDate, LastDate, StartTrain, LastTrain , 
                  Plot = False, Printing = False):
        self.Path = Path
        self.NameStocks = NameStocks
        self.NameSaving = NameSaving
        self.StartDate = StartDate
        self.LastDate = LastDate
        self.StartTrain = StartTrain
        self.LastTrain = LastTrain
        self.Plot = Plot
        self.Printing = Printing
        self.PathNameofStocks = self.Path+'/'+self.NameStocks
        self.PathtoSave = self.Path+'/'+self.NameSaving+'.csv'
        
      
        
        
        
        
        
        

    def KmeansPlot (self,Data):
        LossofKmeans = list()
        for i in range (2 , 11):
            kmeans_=KMeans(n_clusters=i,n_init=1)
            kmeans_=kmeans_.fit(Data)
            LossofKmeans.append(kmeans_.inertia_)
        LenLoss=np.arange(2 , 11)   
        plt.plot(LenLoss, LossofKmeans , label='Loss of Kmeans Method')
        plt.title("Plot for Elbow method")
        plt.grid()
        plt.legend()
        
        
        
    def Cluster (self, Name, NumClusters=5):
        '''
        data should be DataFrame and n_clusters should be int
        but name is char
        '''
        '''
        this for clustring mutual inf and spreat data to n_cluster based on 
        n_cluster and put them in special dataframe'''
        
        ObjectofMutual = mf.MutualInformation (self.Path, self.NameStocks, 
                                               self.NameSaving, self.StartDate, self.LastDate, 
                                               self.StartTrain, self.LastTrain)
        MutualInfData = ObjectofMutual.CalMutualInf()
        NameofStocks , DataTrain, DictofAdj , LenofWeeks = ObjectofMutual.ConverttoDict()
        
        if self.Plot:
            self.KmeansPlot(MutualInfData)
        
        if NumClusters==5:
            warnings.warn('you are using difult K_cluste')
        
        kmeans_=KMeans(NumClusters,n_init=1,random_state=20)
        kmeans_=kmeans_.fit(MutualInfData)
        silh_=silhouette_samples(MutualInfData,kmeans_.labels_)
        newIdx=np.argsort(kmeans_.labels_)
        NewDataFrame=MutualInfData.iloc[newIdx]
        NewDataFrame=NewDataFrame.iloc[:,newIdx]
        uniq = np.unique (kmeans_.labels_)
        DictofClusters  = dict()
        
        for i  in range(len(uniq)):
            globals()[Name+str(uniq[i])]=dict()
            Indexes = [j for j in range(len(NameofStocks)) if kmeans_.labels_[j] == uniq[i]]
            
            for z in range(len(Indexes)):
                globals()[Name+str(uniq[i])][NameofStocks[Indexes[z]]] = \
                DictofAdj[NameofStocks[Indexes[z]]]
            DictofClusters[Name+str(uniq[i])] = globals()[Name+str(uniq[i])]
            
                
        return (DictofClusters)
