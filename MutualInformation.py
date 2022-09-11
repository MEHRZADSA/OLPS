import copy
import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tnrange
import scipy.stats as ss 
from sklearn.metrics import mutual_info_score
import math
import LoadData as ld




class MutualInformation ():
    
    
    def __init__ (self, Path, NameStocks, NameSaving, StartDate, LastDate, StartTrain, LastTrain):
        self.Path = Path
        self.NameStocks = NameStocks
        self.NameSaving = NameSaving
        self.StartDate = StartDate
        self.LastDate = LastDate
        self.StartTrain = StartTrain
        self.LastTrain = LastTrain
        self.PathNameofStocks = self.Path+'/'+self.NameStocks
        self.PathtoSave = self.Path+'/'+self.NameSaving+'.csv'


    def ConverttoDict (self):
        '''
        to be able to use easily data frist we convert dataframe to dict
        '''
        
        LoadingObject = ld.LoadData(self.Path, self.NameStocks, self.NameSaving, self.StartDate, self.LastDate)
        Data = LoadingObject.Loading() # the whole data which downloaded 
        DataTrain = Data.loc[self.StartTrain : self.LastTrain] # the data for testing as dataframe form
        NameofStocks = DataTrain.columns #name of stockes 
        DataTrainCopy = copy.deepcopy(DataTrain) # get copy of Dataframe
        DataTrainCopy = DataTrainCopy.reset_index() 
        DataTrainCopy.drop("Date", axis=1, inplace=True)
        DictofAdj = DataTrainCopy.to_dict('list')# data for testing as dict form
        LenofWeeks = len(DictofAdj)
        return(NameofStocks, DataTrain, DictofAdj, LenofWeeks)
    
    
    def IndexDate (self):
        _,DataTrain ,_ ,_ = self.ConverttoDict()
        Indexs = DataTrain.index
        return(Indexs)

    
    
    def MutualInfFormula(self, X, Y, bins):
        
        
        cXY=np.histogram2d(np.reshape(np.array(X), (len(X))),
                  np.reshape(np.array(Y), (len(X)))
                  ,round(bins))[0]
        Hx=ss.entropy(np.histogram(np.reshape(np.array(X), (len(X))), round(bins))[0])
        Hy=ss.entropy(np.histogram(np.reshape(np.array(Y), (len(X))), round(bins))[0])
        iXY=mutual_info_score(None,None,contingency=cXY)
        iXYn=iXY/min(Hx,Hy)
        hXY=Hx+Hy-iXY 
        return(iXYn)
    
    def CalMutualInf (self):
        '''
        to calculate mutual information number of bins is need
        '''
        NameofStocks , DataTrain , DictofAdj , LenofWeeks = self.ConverttoDict()
        Bin=(8+(32*LenofWeeks)+(12*math.sqrt(36*LenofWeeks+(12*LenofWeeks**2))))**1/3
        MutualMatrix = np.zeros((len(NameofStocks),len(NameofStocks)))
        MutualDataframe = pd.DataFrame(MutualMatrix, columns=NameofStocks, index=NameofStocks)
        for i in tnrange(len(NameofStocks),desc='1st loop'):
            for j in range(len(NameofStocks)):
                a1=NameofStocks[i]
                a2=NameofStocks[j]
                if MutualDataframe[a2][a1] != 0:
                    MutualDataframe[a1][a2] = MutualDataframe[a2][a1]
                else:
                    MutualNum = self.MutualInfFormula(DictofAdj[a1],DictofAdj[a2],Bin)
                    MutualDataframe[a1][a2] = MutualNum
        return(MutualDataframe)
