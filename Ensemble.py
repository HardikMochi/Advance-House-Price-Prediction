
import pandas as pd
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor 
from sklearn.ensemble import GradientBoostingClassifier,StackingRegressor,VotingRegressor
from Regression import Regression

class Ensemble(Regression):
    def __init__(self, ensemble_method,X_train, y_train, X_val, y_val,hyper_tuning_method = "grid"):
        self.ensemble_method = ensemble_method
        self.x_train = X_train
        self.y_train = y_train
        self.x_val = X_val
        self.y_val = y_val
        self.model_type = ensemble_method
        self.hyper_tuning_method = hyper_tuning_method
        self.scores_table = pd.DataFrame()
        self.estimators = None

        if self.ensemble_method == "XGBoost":
            self.model = XGBRegressor(n_jobs=-1)
        elif self.ensemble_method == "LightGbm":
            self.model = CatBoostRegressor()
        elif self.ensemble_method == "Voting":
            self.model = VotingRegressor()  
        
    def Stacking(self,estimators):
        self.model = StackingRegressor(estimators=estimators,final_estimator=RandomForestRegressor(n_estimators=100,random_state=42))
        
        
    def Voting(self,estimators):
        self.model = VotingRegressor(estimators=estimators,final_estimator=RandomForestRegressor(n_estimators=10,random_state=42))


    

