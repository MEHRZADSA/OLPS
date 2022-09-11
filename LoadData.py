
import pandas as pd
import datetime as dt
import DownloadData as dd
class LoadData ():
   def __init__ (self, Path, NameStocks, NameSaving, StartDate, LastDate):
    '''
    Path :The path uses for save data or load data from the place
    NameStocks: is a csv file which the name of stock is saved in it
    NameSaving: is a csv file taht will be created and data values  saves in that
    StartDate: First day of Data which is neede
    LastDate: Last day of Data for project 
    
    '''
    self.Path = Path
    self.NameStocks = NameStocks
    self.NameSaving = NameSaving
    self.SartDate = StartDate
    self.LastDate = LastDate
    self.PathNameofStocks = self.Path+'/'+self.NameStocks
    self.PathtoSave = self.Path+'/'+self.NameSaving+'.csv'
    
    
   def Loading (self):
    try:
        DictofStocks = pd.read_csv(self.PathtoSave)
        DictofStocks["Date"]=DictofStocks["Date"].apply(lambda X: dt.datetime.strptime(X,"%Y-%m-%d").date())
        DictofStocks.set_index('Date', inplace=True)
        return(DictofStocks)
    except:
        Saving = dd.DownloadData(self.PathNameofStocks)
        Saving.SaveOnDisc(self.NameSaving, self.SartDate, self.LastDate)
        DictofStocks = pd.read_csv(self.PathtoSave)
        DictofStocks["Date"]=DictofStocks["Date"].apply(lambda X: dt.datetime.strptime(X,"%Y-%m-%d").date())
        DictofStocks.set_index('Date', inplace=True)
        return(DictofStocks)
   def IndexDate ():
    DictofStocks = self.Loading
    Indexs = DictofStock.index
    return(index)
