from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.ensemble import StackingRegressor


# Define Models
# XGB Regressor
xgb_params = dict(max_depth= 4,
                  learning_rate= 0.005005070416155941,
                  n_estimators= 7650,
                  min_child_weight= 2,
                  colsample_bytree= 0.20263034530849983,
                  subsample= 0.4402289758648288,
                  reg_alpha= 0.0010309970136600966,
                  reg_lambda= 0.012884368300273313,
                  random_state=1)
                  

xgb =  XGBRegressor(**xgb_params)


# ElasticNet
elasticnet = ElasticNet(max_iter=1000,
                        alpha=0.0007,
                        l1_ratio=0.9,
                        random_state=1)


# Create Stacking model for improved prediction
base_models = [
    ('elasticnet', elasticnet),
    ('xgb', xgb)
]


meta_model = LinearRegression()
stack = StackingRegressor(estimators=base_models, final_estimator=meta_model, cv = 5)


