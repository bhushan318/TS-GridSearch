import pandas as pd
import numpy as np 
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
from collections import deque
import joblib
import datetime


class StreetPricePredictor:
    ''' A simple class to easily query the ETS model from inside AnyLogic using Pypeline '''
    def __init__(self):
        # load both policies
        self.ets_model = joblib.load('model.pkl')
        self.start_date = datetime.datetime(2020, 9, 12)
        
        # initialize queue with random values to feed into neural network model
        # note 1: these values will be in range [-1, 1]
        # note 2: the model was trained to predict the next value from last *6*
        # (each sample represents the hourly arrive rate for a 4 hour time span)
        init_values = np.random.random((6,))*2-1
        self.last_rates_queue = deque(init_values, maxlen=6)


    def predict_price(self, patient_data):
        ''' Given the (1, 24) array of patient data, predict the length of stay (days) '''
        
        # convert default list to numpy array
        patient_array = pd.to_datetime(self.start_date+datetime.timedelta(days=7*patient_data))
        
        # query the neural network for the length of stay
        prediction =  self.ets_model.predict(patient_array)
        return prediction.values[0]