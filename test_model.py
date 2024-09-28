# Mute warnings
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd

# import raw test data
test_data = pd.read_csv('data/test.csv', index_col="Id")

# load model from joblib file
with open(f'data/model.joblib','rb') as file:
    model = joblib.load(file)
    
# run inference on test data
predictions = np.exp(model.predict(test_data))
predictions = np.round(predictions, 2)
output = pd.DataFrame({'Id': test_data.index, 'SalePrice': predictions})
output.to_csv('y_pred.csv', index=False)
print('predictions:\n', output.head())

# run prediction on a few samples
test_data2 = test_data.iloc[0:5].values
predictions = np.round(np.exp(model.predict(test_data2)), 2)
print('\npredictions:\n', predictions)