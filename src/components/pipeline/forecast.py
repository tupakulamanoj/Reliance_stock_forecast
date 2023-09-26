import sys
from src.logger import logging
from src.exception import custom_exception
import pickle
import pandas as pd
from datetime import datetime,timedelta
class forecast:
    def forecasting_data(self,days):
        logging.info('data forecasting has started')
        close = []
        high = []
        low = []
        open_prices = []
        volume = []
        columns = ['Close', 'Open', 'High', 'Volume', 'Low']
        
        start_date = datetime.now()
        dat = start_date + timedelta(days=days)
        date = pd.date_range(start_date, dat)
        day = date.day
        month = date.month
        year = date.year
        val = pd.DataFrame({'year': year, 'day': day, 'month': month})
        
        for fore_column in columns:
            print(fore_column)
            with open(f'Artifacts\{fore_column}model.pkl', 'rb') as file:
                model = pickle.load(file)
            forecast_data = model.predict(val)
            
            if fore_column == 'Close':
                close = forecast_data
            elif fore_column == 'Open':
                open_prices = forecast_data
            elif fore_column == 'Volume':
                volume = forecast_data
            elif fore_column == 'High':
                high = forecast_data
            else:
                low = forecast_data

        fore_data = pd.DataFrame({'Date': date, 'Open': open_prices, 'Low': low, 'High': high, 'Close': close, 'Volume': volume})
            
        logging.info(f'forecast data for reliance industries for {days} is {fore_data}')
        
        return date,open_prices,low,high,close,volume
obj=forecast()
obj.forecasting_data(10)