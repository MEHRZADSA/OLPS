import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tnrange

from sklearn.metrics import mutual_info_score
#from PyCausality.TransferEntropy import TransferEntropy
import scipy.stats as ss 
import scipy
import math
import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import warnings
warnings.filterwarnings('ignore')


import DownloadData as dd
import LoadData as ld
import MutualInformation as mf
import Clustering as cl
import FindingOptimalNum as Fo








class ForecastingNextDay ():
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
        
        
    def ChangeToReturn (self , Name='RCluster'):
        ObjectMI = mf.MutualInformation (self.Path, self.NameStocks, 
                                         self.NameSaving, self.StartDate, 
                                         self.LastDate, self.StartTrain, self.LastTrain)
        
        Indexes = ObjectMI.IndexDate()
        
        
        ObjectCl = cl.Clustering(self.Path, self.NameStocks, 
                                 self.NameSaving, self.StartDate, 
                                 self.LastDate, self.StartTrain, self.LastTrain)
        
        DictofClusters = ObjectCl.Cluster("Cluster")
        
        def ChangeDataFrame (X):
            Variable =  pd.DataFrame(X)
            Variable = Variable.set_index(Indexes)
            return (Variable)
        ListOfKeys = list(DictofClusters.keys())
        DictofDataFramesR = dict()
        DictofDataFramesP = dict()
        for i in range (len (ListOfKeys)):
            DictofDataFramesR ['RCluster'+str(i)] = ChangeDataFrame(DictofClusters['Cluster'+str(i)]).pct_change().dropna(how='all')
            DictofDataFramesP  ['PCluster'+str(i)] = ChangeDataFrame(DictofClusters['Cluster'+str(i)])
        # one of the is dataframe of prices and other is dataframe of returns
        return (DictofDataFramesR, DictofClusters , DictofDataFramesP, ListOfKeys)
    
    
    
    
    def ToData(self , data, lag):
        EmptyDataFrame = pd.DataFrame()
        EmptyDataFrames = pd.DataFrame(columns=data.columns)
    
        for i in np.arange(0,len(data)-lag,1):
            EmptyDataFrame = EmptyDataFrame.append(pd.Series(np.reshape(data.iloc[i:i+lag,:].to_numpy(),
                                                                    (-1))),ignore_index=True)
            if i < len(data)-1 :
                EmptyDataFrames = EmptyDataFrames.append(data.iloc[i+lag,:])
            else:
                EmptyDataFrames = EmptyDataFrames.append(data.iloc[i+1,:])
        EmptyDataFrame = EmptyDataFrame.set_index(data.index[:-lag])
        return((EmptyDataFrame,EmptyDataFrames))
    
    
    
    
    def Forecast(self, Data, lag, number_of_cluster):
        
        Data = Data.pct_change().dropna(how='all') # change to return as well
        data_for_clustring,_ = self.To_Data(Data,lag)
        kmeans_ = KMeans(number_of_cluster ,n_init=1 ,random_state=1)#number of cluster which wae founded
        kmeans_ = kmeans_.fit(data_for_clustring)
    
        for i in range(number_of_cluster):#number of cluster
            globals()['majazi'+str(i)]=pd.DataFrame()
        
        for i,j in enumerate(kmeans_.labels_):
            globals()['majazi'+str(j)]=globals()['majazi'+str(j)].append(data_for_clustring.iloc[i,:])
        
        jj=pd.DataFrame()
        jj=jj.append(pd.Series(np.reshape(data.iloc[-lag:,:].to_numpy(),(-1))),ignore_index=True)
        predict=kmeans_.predict(jj)[0]
        classe=globals()['majazi'+str(predict)]
        cor_dic={}
        cor_list=[]
    
        for i,j in enumerate(classe.index):
            cor_dic[j]=np.corrcoef(classe.loc[j],jj)[0][1]
            cor_list.append(np.corrcoef(classe.loc[j],jj)[0][1])
        
        for i in (classe.index):
            cor_dic[i]=cor_dic[i]/np.sum(cor_list)
        
        data_for_tom=pd.DataFrame()
        index=[]
        for i in classe.index:
            index.append(i+dt.timedelta(lag*7))
        data_for_tom=data.loc[index]
    
        for i,j in enumerate(data_for_tom.index):
            data_for_tom.loc[j]=data_for_tom.loc[j]*cor_list[i]
        
        forecasted=np.sum(data_for_tom)
        aj=(pd.DataFrame(pd.Series(forecasted)).T)
        aj.set_index='tom'
        return(aj)

    def Forecasting (self):
    
        ObjectofFo = Fo.FindingOptimalNum (self.Path, self.NameStocks, 
                                       self.NameSaving, self.StartDate, 
                                       self.LastDate, self.StartTrain, self.LastTrain )
    
    
        DictofDataFramesR, _ , DictofDataFramesP, ListOfKeys = self.ChangeToReturn ()
        DictofLags , DictofNmuberofCluster = ObjectofFo.CalBestParams()
    
        for i in range(len (ListOfKeys)):
            Predicted = for_forcasting(globals()['ClusterForMutual'+str(i)],
                DictofLags['lag'+str(i)],
            DictofNmuberofCluster ['number_of_best_clustr'+str(i)])
            lag = DictofLags['lag'+str(i)]
            globals()['data_for_portfolio'+str(i)]  = DictofDataFramesR['RCluster'+str(i)].\
            iloc[-lag:,:]
            globals()['data_for_portfolio'+str(i)] = globals()['data_for_portfo'+str(i)].append(Predicted)
