import os
import sys 
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.exception import custom_exception
from src.components.model_training import model_training_testing
@dataclass
class data_config_path:
    logging.info('Data path created for Artifacts1')
    data_path=os.path.join('Artifacts','forecast_data.csv')
class data_ingest_transform:
    def __init__(self):
        self.intial_data_path=data_config_path()
    def data_ingestion_transformation(self):
            logging.info('Data read sucessfully!')
            data=pd.read_csv(r'C:\Users\manoj\Desktop\pride_project\data\reliance_data.csv')
            os.makedirs(os.path.dirname(self.intial_data_path.data_path),exist_ok=True)
            logging.info('data ingestion sucessfully!')
            logging.info('data transformation started!')
            df=data.drop(['Unnamed: 0'],axis=1)
            df.dropna(inplace=True)
            df.reset_index(drop=True,inplace=True)
            lis=['Close',"Open","High","Low",'Volume']
            year=[]
            month=[]
            day=[]
            for i in range(len(df)):
                year.append(df.Date[i].split('-')[0])
                month.append(df.Date[i].split('-')[1])
                day.append(df.Date[i].split('-')[2])
            df['year']=[int(i) for i in year]
            df['day']=[int(i) for i in day]
            df['month']=[int(i) for i in month]
            logging.info('data transformation sucessfully completed')
            df.to_csv(os.path.join(self.intial_data_path.data_path),header=True,index=False)
            logging.info('data is ready for model training')
            return self.intial_data_path.data_path
                    
        
if __name__=="__main__":
    obj=data_ingest_transform()
    data_path=obj.data_ingestion_transformation() 
    obj2=model_training_testing()
    obj2.model_trainig(data_path)