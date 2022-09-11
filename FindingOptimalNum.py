import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tnrange
import scipy.stats as ss
import optuna
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import warnings
import DownloadData as dd
import LoadData as ld
import MutualInformation as mf
import Clustering as cl












class FindingOptimalNum ():
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
        
    def Optimizer(self,df):
        '''
        first : we cereat an object due to train our model and get our loss
        second : we calculate our loss by ignoring many days 
        the best combination will be chosing for our model
        '''
        
        def objective(trial, lag_min=2, lag_max=15, num_min=2, num_max=15):
            lag_= trial.suggest_int("lag", lag_min, lag_max)
            number_cluster_=trial.suggest_int("number_cluster_", num_min, num_max)
            return self.CalNextDay(df,lag_,number_cluster_)
        
        
        lag_min = 2
        lag_max = 15

        num_min=2
        num_max=15
        
        
        search_space = {"lag": range(2,15), "number_cluster_": range(2,15)}
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(sampler=optuna.samplers.GridSampler(search_space))
        # Execute an optimization by using the above objective function wrapped by `lambda`.
        study.optimize(lambda trial: objective(trial, lag_min, lag_max,num_min,num_max), n_trials=100)
        return(study.best_params)
    
    
    def ToData(self , data, lag):
        EmptyDataFrame = pd.DataFrame()
        EmptyDataFrames = pd.DataFrame(columns=data.columns)
    
        for i in np.arange(0,len(data)-lag,1):
            EmptyDataFrame = EmptyDataFrame.append(pd.Series(np.reshape(data.iloc[i:i+lag,:].to_numpy(),
                                                                    (-1))),ignore_index=True)
            if i <103:
                EmptyDataFrames = EmptyDataFrames.append(data.iloc[i+lag,:])
            else:
                EmptyDataFrames = EmptyDataFrames.append(data.iloc[i+1,:])
        EmptyDataFrame = EmptyDataFrame.set_index(data.index[:-lag])
        return((EmptyDataFrame,EmptyDataFrames))


    def Loss (self ,Real,Predicted):
        Lost=[]
        for i in range(len(Real)):
            Diverted = np.abs(Real[i]-Predicted[i])
            Lost.append(Diverted)
            
        return(np.sum(Lost))

    
    
    def CalNextDay (self ,Data,lag,number_cluster):
    
        Data=Data.pct_change().dropna(how='all')
        DataTarin=Data.iloc[:-1,:]#data for split
        DataTest=Data.iloc[-1]
        DataforClustering,_ = self.ToData(DataTarin, lag)
        kmeans_ = KMeans(number_cluster,n_init=1,random_state=1)
        kmeans_ = kmeans_.fit(DataforClustering)
        for i in range(number_cluster): #creat permanent Variable for testing 
            globals()['majazi'+str(i)]=pd.DataFrame()
        
        for i,j in enumerate(kmeans_.labels_):
            globals()['majazi'+str(j)]=globals()['majazi'+str(j)].append(DataforClustering.iloc[i,:])
        
        NextDay = pd.DataFrame()
        NextDay = NextDay.append(pd.Series(np.reshape(DataTarin.iloc[-lag:,:].to_numpy(),(-1))),
                             ignore_index=True)
        predict=kmeans_.predict(NextDay)[0]
        classe=globals()['majazi'+str(predict)]
        cor_dic={}
        cor_list=[]
        for i,j in enumerate(classe.index):
            cor_dic[j]=np.corrcoef(classe.loc[j],NextDay)[0][1]
            cor_list.append(np.corrcoef(classe.loc[j],NextDay)[0][1])
        
        for i in (classe.index):
            cor_dic[i]=cor_dic[i]/np.sum(cor_list)
        DataNext = pd.DataFrame() #datafor tom
        index=[]
        for i in classe.index:
            index.append(i+dt.timedelta(lag*7))
        
        DataNext = DataTarin.loc[index]
        for i,j in enumerate(DataNext.index):
            DataNext.loc[j] = DataNext.loc[j]*cor_list[i]
        forecasted = np.sum(DataNext)
        losss = self.Loss(forecasted,DataTest)
        return(losss)

    def CalBestParams (self):
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
        '''    
        with concurrent.futures.ThreadPoolExecutor() as executor:
            
            for i in range (len(DictofClusters.keys())):
                globals()['f'+str(i)] = executor.submit (self.Optimizer ,
                                                    ChangeDataFrame (DictofClusters[ListOfKeys[i]]))
                
            for i in range (len(DictofClusters.keys())):
                print (globals()['f'+str(i)])
                

        '''   
        DictofLags = dict()
        DictofNmuberofCluster = dict()
        for i in range(len(DictofClusters)):
            
            Vari0 =ChangeDataFrame(DictofClusters['Cluster'+str(i)])
            Vari1 = self.Optimizer(Vari0)
            DictofLags['lag'+str(i)] = Vari1['lag']
            DictofNmuberofCluster['number_of_best_clustr'+str(i)] = Vari1['number_cluster_']
            
        return (DictofLags , DictofNmuberofCluster)
