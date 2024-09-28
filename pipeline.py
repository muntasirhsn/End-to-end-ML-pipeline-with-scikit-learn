# Mute warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

# Import custom classes and functions from preprocessing.py
from preprocessor import MedianNeighborhoodTransformer, data_preprocessor, scaler, ohe, FeatureNamer, column_transformer

# Import stack model
from models import stack


# Import train data
train = pd.read_csv(f"data/train.csv", index_col="Id")
X_train = train.copy()
print('Train data size: ', X_train.shape)
y_train = X_train.pop("SalePrice")

feature_names = X_train.columns.tolist()
print('total number of initial features: ', len(feature_names))


# Full pipeline
pipeline_model = Pipeline(steps=[
    ('feature_namer', FeatureNamer(feature_names)),  # Automatically assigns feature names if missing
    ('data_preprocessor', data_preprocessor),
    ('transform', column_transformer),
    ('stack_model', stack)
])


######################################
# Optional: Evaluate pipeline model with Cross-Validation 
# Root Mean Squared Logaritmic Error (RMSLE)
def rmsle(X, y, model):
    X = X.copy()
    score = -cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    score = score.mean()
    score = np.sqrt(score)
    return score

# Due to to the nature of the traget feature of the training data, we fit the model to the logarithmic value of the target
print('\nModel evaluation in progress....')
score = rmsle(X_train, np.log1p(y_train), pipeline_model)
print("Root mean squared logarithmic (rmsle) error: {:.5f}".format(score), '\n')
#####################################

# Fit the pipeline to training data
# Due to the nature of the traget feature of the training data, we fit the model to the logarithmic value of the target
print('Fitting pipeline on full training data')
pipeline_model.fit(X_train, np.log1p(y_train))
print("Pipeline model training complete!")


# Save the pipeline using joblib
joblib.dump(pipeline_model, 'data/model.joblib') # make sure directory 'data' exits prior to executing code
print("Pipeline model saved as a joblib file (data/model.joblib)")

