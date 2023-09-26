import sys
import pandas as pd
from src.exception import custom_exception
from src.logger import logging
import pickle
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import warnings
from  datetime import datetime,timedelta
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
logging.info('All modules are sucessfully imported')



class model_training_testing:
    def model_trainig(self,data_path):
        try:
            df=pd.read_csv(data_path)
            logging.info('data read sucessfully')
            lis=['Close',"Open","High","Low",'Volume']
            for column in lis:
                x=df.iloc[:,-3:]
                y=df[column]
                logging.info(f'{column} model testing started')
                x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
                logging.info(f'{column} data splitted sucessfully')
                models={'LinearRegression':LinearRegression(),
                        "RandomForestRegressor":RandomForestRegressor(),
                        'SVR':SVR(),
                        "DecisionTreeRegressor":DecisionTreeRegressor(),
                        "ExtraTreeRegressor":ExtraTreeRegressor()
                        ,"XGBRegressor":XGBRegressor(),
                        "XGBRFRegressor":XGBRFRegressor(),
                        "GradientBoostingRegressor":GradientBoostingRegressor(),
                        "AdaBoostRegressor":AdaBoostRegressor(),
                        "KNeighborsRegressor":KNeighborsRegressor()
                        }
                
                para={"LinearRegression":{},
                    "RandomForestRegressor":{'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'max_features':['sqrt','log2',None],
                        'n_estimators': [8,16,32,64,128,256]},
                    'SVR': {'kernel': ['linear', 'rbf', 'poly'],'C': [1, 10, 100],'gamma': [0.1, 1, 'auto'],'epsilon': [0.1, 0.01, 0.001]},
                    'DecisionTreeRegressor': {
                        'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        'splitter':['best','random'],
                        'max_features':['sqrt','log2'],},
                    'ExtraTreesRegressor':{},
                    "XGBRegressor":{
                        'learning_rate':[.1,.01,.05,.001],
                        'n_estimators': [8,16,32,64,128,256]
                    },
                    "XGBRFRegressor":{
                            'n_estimators': [50, 100, 200],
                            'max_depth': [3, 4, 5],
                            'learning_rate': [0.1, 0.01, 0.001],
                            'subsample': [0.8, 0.9, 1.0],
                            'colsample_bynode': [0.8, 0.9, 1.0],
                        },
                    "GradientBoostingRegressor":{ 'n_estimators': [50, 100, 200],
                                                    'learning_rate': [0.1, 0.01, 0.001],
                                                    'max_depth': [3, 4, 5],
                                                    'min_samples_split': [2, 5, 10],
                                                    'min_samples_leaf': [1, 2, 4],},"AdaBoostRegressor":{
                                                    'n_estimators': [50, 100, 200],
                                                    'learning_rate': [0.1, 0.01, 0.001],
                                                },
                    'KNeighborsRegressor':{
                        'n_neighbors': [3, 5, 7, 9],
                        'weights': ['uniform', 'distance'],
                        'p': [1, 2],
                    },
                    }
                logging.info(f"hyper tuning started for {column} data")
                def model_accuracy(x_train,y_train,x_test,y_test,models,para):
                    mo=[]
                    po=[]
                    for i in  models.values():
                        mo.append(i)
                    report={}
                    for j in para.values():
                        po.append(j)
                        
                    for mod in range(len(mo)):
                        model=mo[mod]
                        para=po[mod]
                        logging.info('grid search cv is started')
                        gs=GridSearchCV(model,param_grid=para,cv=3)
                        gs.fit(x_train,y_train)
                        
                        model.set_params(**gs.best_params_)
                        
                        model.fit(x_train,y_train)
                        pred=model.predict(x_test)
                        report[model]=r2_score(y_test,pred)
                    max_accuracy=max(list(report.values()))
                    for i,j in report.items():
                        if  j == max_accuracy:  
                            model=i
                    logging.info(f'algorithm {report}')
                    logging.info(f'best model for data forecasting {model}')
                    return report,model
                report,model=model_accuracy(x_train,y_train,x_test,y_test,models,para)
                model.fit(x_train,y_train)
                y_pred=model.predict(x_test)
                logging.info('pickle file created for best model')
                with open(f'Artifacts\{column}model.pkl','wb') as file:
                    pickle.dump(model,file)
        except Exception as e:
            raise custom_exception(e,sys)
            