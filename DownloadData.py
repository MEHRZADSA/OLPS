import pandas as pd
import numpy as np
from pandas_datareader.data import DataReader
import datetime as dt
from tqdm import tnrange
import ipywidgets as widgets
from ipywidgets import interact, fixed
import math
from tqdm import tnrange
import warnings
warnings.filterwarnings('ignore')
class DownloadData :
    
   def __init__(self,Path):
    """
    the path is needed to read the stock names
    
    """
    self.Path = Path
    
   def ReadStockNames(self):
        """
        the stock names is saved as csv file
        
        
        due to quotation marks which is added unwanted the RemoveQu is used
        
        """
        StockNames = pd.read_csv(self.Path+".csv",header=None)
        StockNames = StockNames.iloc[:,0].values.tolist()
        
        RemoveQu = lambda y :y[1:-1]
        StockNames = list(map(RemoveQu,StockNames))
        
        return (StockNames[:10])
    
    
   def ReadDataOnline (self, StartDate , LastDate):
    """
    1.Run previous method
    2.Generate empty DataFrame
    3.Generet Name for each stock and give each name its value
    
    StartDate: First day of Data which is neede
    LastDate: Last day of Data for project
    """
    StockNames = self.ReadStockNames()
    DictofStocks = pd.DataFrame()
    
    for i in tnrange(len(StockNames)):
        try:
            Name = StockNames[i]
            globals()[Name] = DataReader(Name, 'yahoo',StartDate,LastDate)['Adj Close']
            DictofStocks[Name] = globals()[Name]
        except:
            pass
    return(DictofStocks)

     
    
   def ChangetoWeekly(self, StartDate , LastDate): 
    """
    change data to weekly insist of daily
    """
    DictofStocks = self.ReadDataOnline(StartDate , LastDate)
    DictofStocks = DictofStocks.resample('W').mean()
    return(DictofStocks)
 
    
    
    
    
    

   def SaveOnDisc (self, Name , StartDate , LastDate):
    """
    data should be saved on disc, for its downloading taks too long
    
    Name: must be string
    
    StartDate: First day of Data which is neede
    
    LastDate: Last day of Data for project
    
    """
    try:
        DataFrameofStocks = self.ChangetoWeekly(StartDate , LastDate)
    
        DataFrameofStocks.to_csv(Name+'.csv')
        
        print('your data downloaded successfully')
    except:
        print ('Something went wrong ')