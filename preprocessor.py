import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, RobustScaler


# Renaming some features. Feature names beginning with numbers are awkward to work with, so we rename them
def rename_columns(df):
    X = df.copy()
    # Names beginning with numbers are awkward to work with
    X.rename(columns={"1stFlrSF": "FirstFlrSF",
                       "2ndFlrSF": "SecondFlrSF",
                       "3SsnPorch": "Threeseasonporch"},
              inplace=True)
    return X


# Clean Data
def clean(df):
    X = df.copy()
    X["Exterior2nd"] = X["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})    
    # Some values of GarageYrBlt are corrupt, so we'll replace them with the year the house was built
    X["GarageYrBlt"] = X["GarageYrBlt"].where(X.GarageYrBlt <= 2010, X.YearBuilt)    
    # Some values of YrSold are also corrupt i.e. house was sold before it was built! We'll replace them with the year the house was built
    X["YrSold"] = X["YrSold"].where(X.YrSold >= X.YearBuilt, X.YearBuilt)    
    return X


# The 'MSSubClass' is a string/unordered(nominal) categorical variable but encoded as `int` type. So we need to convert the data type. 
other_categorical_features = ['MSSubClass'] 
def numer_to_object(df):
    X = df.copy()
    for name in other_categorical_features:
        X[name] = X[name].astype('str')
    return X


# Handle Missing Values
def impute(df):
    X = df.copy()
    for name in X.select_dtypes("number"):
        X[name] = X[name].fillna(0)
    for name in X.select_dtypes("O"):
        X[name] = X[name].fillna("None") 
    for name in X.select_dtypes("category"):
        X[name] = X[name].fillna("None") 
    return X


# Categorical features: ordinal encoding
# Some of the categorical features are ordered so we need to correctly encode them with specific orders
ordered_levels = {
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "PoolQC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# add a new category 'None' in each ordinal categorical feature so that we can replace NAs with a category 'None'.
ordered_levels = {key: ["None"] + value for key, value in ordered_levels.items()} # the 1st level is 'None'

def ordinal_encode(df):
    X = df.copy()
    # Ordinal categories
    for name, levels in ordered_levels.items():
        X[name] = X[name].astype(CategoricalDtype(levels, ordered=True)) 
    return X


# Categorical features: nominal encoding
unordered_levels = {'MSSubClass': ['None', '60', '20', '70', '50', '190', '45', '90', '120', '30', '85', '80', '160', '75', '180', '40'], 
          'MSZoning': ['None', 'RL', 'RM', 'C (all)', 'FV', 'RH'], 
          'Street': ['None', 'Pave', 'Grvl'], 
          'Alley': ['None', 'Grvl', 'Pave'], 
          'LandContour': ['None', 'Lvl', 'Bnk', 'Low', 'HLS'], 
          'LotConfig': ['None', 'Inside', 'FR2', 'Corner', 'CulDSac', 'FR3'], 
          'Neighborhood': ['None', 'CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste'], 
          'Condition1': ['None', 'Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe'], 
          'Condition2': ['None', 'Norm', 'Artery', 'RRNn', 'Feedr', 'PosN', 'PosA', 'RRAn', 'RRAe'], 
          'BldgType': ['None', '1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], 
          'HouseStyle': ['None', '2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'], 
          'RoofStyle': ['None', 'Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'], 
          'RoofMatl': ['None', 'CompShg', 'WdShngl', 'Metal', 'WdShake', 'Membran', 'Tar&Grv', 'Roll', 'ClyTile'], 
          'Exterior1st': ['None', 'VinylSd', 'MetalSd', 'Wd Sdng', 'HdBoard', 'BrkFace', 'WdShing', 'CemntBd', 'Plywood', 'AsbShng', 'Stucco', 'BrkComm', 'AsphShn', 'Stone', 'ImStucc', 'CBlock'], 
          'Exterior2nd': ['None', 'VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng', 'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'BrkComm', 'ImStucc', 'AsphShn', 'Stone', 'Other', 'CBlock'], 
          'MasVnrType': ['BrkFace', 'None', 'Stone', 'BrkCmn'], 
          'Foundation': ['None', 'PConc', 'CBlock', 'BrkTil', 'Wood', 'Slab', 'Stone'], 
          'Heating': ['None', 'GasA', 'GasW', 'Grav', 'Wall', 'OthW', 'Floor'], 
          'GarageType': ['Attchd', 'Detchd', 'BuiltIn', 'CarPort', 'None', 'Basment', '2Types'], 
          'MiscFeature': ['None', 'Shed', 'Gar2', 'Othr', 'TenC'], 
          'SaleType': ['None', 'WD', 'New', 'COD', 'ConLD', 'ConLI', 'CWD', 'ConLw', 'Con', 'Oth'], 
          'SaleCondition': ['None', 'Normal', 'Abnorml', 'Partial', 'AdjLand', 'Alloca', 'Family']}

def nominal_encode(df):
    X = df.copy()
    # Ordinal categories
    for name, levels in unordered_levels.items():
        X[name] = X[name].astype(CategoricalDtype(levels, ordered=False))
    return X


# Factorize ordered categorical features with label encoding
def label_encode(df):
    X = df.copy()
    # The `cat.codes` attribute holds the category levels.
    for colname in X.select_dtypes(["category"]).columns:
        if X[colname].cat.ordered:  # Check if the categorical column is ordered
            X[colname] = X[colname].cat.codes
    return X


# Create features with pandas
def create_features(df):    
    X = df.copy()
    
    # mathematical_transforms:
    X["LivLotRatio"] = X["GrLivArea"] / X["LotArea"]
    X["Spaciousness"] = (X["FirstFlrSF"] + X["SecondFlrSF"]) / X["TotRmsAbvGrd"]
    X['NewHouse_RecentRemodel'] = 2*(X['YearBuilt']/2010) + (X['YearRemodAdd'] - X['YearBuilt'])/2010 
                                  # check the plot for 'SAlePrice' vs 'YearBuilt' and 'YearRemodAdd' in section B        
    X['TotalSF'] = X['TotalBsmtSF'] + X['FirstFlrSF'] + X['SecondFlrSF'] 
    X['TotalSF2'] = X['BsmtFinSF1'] + X['BsmtFinSF2'] + X['FirstFlrSF'] + X['SecondFlrSF']
    X['TotalBathrooms'] = X['FullBath'] + (0.5*X['HalfBath']) + X['BsmtFullBath'] + (0.5*X['BsmtHalfBath'])
    X['TotalPorchArea'] = X['OpenPorchSF'] + X['Threeseasonporch'] + X['EnclosedPorch'] + X['ScreenPorch'] + X['WoodDeckSF']    
    X["OverallGrade"] = np.sqrt(X["OverallQual"] * X["OverallCond"])
    X["GarageGrade"] = np.sqrt(X["GarageQual"] * X["GarageCond"])
    X["ExterGrade"] = np.sqrt(X["ExterQual"] * X["ExterCond"])
    
    # special features
    X['HasPool'] = X['PoolArea'].apply(lambda x: 1 if x > 0 else 0)                                                                          
    X['Has2ndfloor'] = X['SecondFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    X['HasGarage'] = X['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    X['HasBasement'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    X['HasFireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    X["HasShed"] = (X["MiscFeature"] == "Shed") * 1 
    X["RemodelbeforeSold"] = (X["YearRemodAdd"] == X["YrSold"])*1  # True(1) if a remodelling happened in the same year the house was sold
    
    X.loc[X.Neighborhood == 'NridgHt', "GoodNeighborhood"] = 1
    X.loc[X.Neighborhood == 'Crawfor', "GoodNeighborhood"] = 1
    X.loc[X.Neighborhood == 'StoneBr', "GoodNeighborhood"] = 1
    X.loc[X.Neighborhood == 'Somerst', "GoodNeighborhood"] = 1
    X.loc[X.Neighborhood == 'NoRidge', "GoodNeighborhood"] = 1
    X["GoodNeighborhood"] = X["GoodNeighborhood"].fillna(0)

    X["AbnormalSaleCondition"] = X.SaleCondition.replace({'Abnorml': 1, 'Alloca': 1, 'AdjLand': 1, 'Family': 1, 'Normal': 0, 'Partial': 0}) 
    X["PartialSale"] = X.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, "Family" : 0, "Normal" : 0, "Partial" : 1}) 
    X["GoodHeating"] = X.HeatingQC.replace({'Ex': 1, 'Gd': 1, 'TA': 0, 'Fa': 0, 'Po': 0})

    area_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
                 'TotalBsmtSF', 'FirstFlrSF', 'SecondFlrSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 
                 'OpenPorchSF', 'EnclosedPorch', 'Threeseasonporch', 'ScreenPorch', 'LowQualFinSF', 'PoolArea']
    X["TotalHouseArea"] = X[area_cols].sum(axis=1) 

    X["TotalArea1st2nd"] = X["FirstFlrSF"] + X["SecondFlrSF"]
    X["HouseAge"] = 2010 - X["YearBuilt"]
    X['SoldAge'] = X.YrSold - X.YearBuilt
       
    neighborhood = {"MeadowV" : 0,  "IDOTRR" : 1, "BrDale" : 1, "OldTown" : 1, "Edwards" : 1, "BrkSide" : 1, "Sawyer" : 1, 
                    "Blueste" : 1, "SWISU" : 2, "NAmes" : 2,  "NPkVill" : 2, "Mitchel" : 2, "SawyerW" : 2, "Gilbert" : 2, 
                    "NWAmes" : 2, "Blmngtn" : 2, "CollgCr" : 2, "ClearCr" : 3, "Crawfor" : 3, "Veenker" : 2, "Somerst" : 3, 
                    "Timber" : 3, "StoneBr" : 3, "NridgHt" : 3, "NoRidge" : 4}

    X["NeighborhoodMap"] = X["Neighborhood"].map(neighborhood)    

    # logarithmic features from highly skewed features
    X["GrLivArea_log"] = np.log1p(X.GrLivArea)
    X['MasVnrArea_log'] = np.log1p(X.MasVnrArea)

    # interaction features
    # Replace pd.get_dummies with a check
    if 'BldgType_Interaction_None' not in X.columns:  # Check against existing columns
        X1 = pd.get_dummies(X.BldgType, prefix="BldgType_Interaction")
        X1 = X1.mul(X.GrLivArea, axis=0)
        X = pd.concat([X, X1], axis=1)
        
    # Replace pd.get_dummies with a check
    if 'Neighborhood_Interaction_None' not in X.columns:  # Check against existing columns
        X2 = pd.get_dummies(X.Neighborhood, prefix="Neighborhood_Interaction")
        X2 = X2.mul(X.GrLivArea, axis=0)
        X = pd.concat([X, X2], axis=1)
    
    # counts features
    X["PorchTypes"] = X[[
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "Threeseasonporch",
        "ScreenPorch",
    ]].gt(0.0).sum(axis=1)

  
    return X


# create feature from median values of "GrLivArea" for each neighborhood 
class MedianNeighborhoodTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = None

    def fit(self, X, y=None):
        # Compute median values of "GrLivArea" for each neighborhood using the training data and store them
        self.medians = X.groupby('Neighborhood')['GrLivArea'].median()
        return self

    def transform(self, X):
        X = X.copy()
        # Apply the precomputed medians from training data to both train/test
        X['MedNhbdArea'] = X['Neighborhood'].map(self.medians)
        return X

# Custom Transformer for data processing
class DataPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.numeric_cols = None
        self.categorical_cols = None
        self.median_transformer = MedianNeighborhoodTransformer()

    def fit(self, X, y=None):
        # Fit the median transformer
        self.median_transformer.fit(X)
        X_transformed = self.transform(X)
        self.numeric_cols = X_transformed.select_dtypes(include=['number']).columns.tolist()
        self.categorical_cols = X_transformed.select_dtypes(exclude=['number']).columns.tolist()
        
        return self

    def transform(self, X):
        X = X.copy()
        # rename columns
        X = rename_columns(X)
    
        # clean X
        X = clean(X)
    
        # nominal (unordered categorical) encode X
        X = numer_to_object(X)
    
        # impute NAs
        X = impute(X)
    
        # ordinal (ordered categorical) encode X
        X = ordinal_encode(X)
    
        # nominal (unordered categorical) encode X
        X = nominal_encode(X)
    
        # factorize ordered categorical features
        X = label_encode(X)
    
        # create features with pandas
        X = create_features(X)
        
        # Apply median transformer
        X = self.median_transformer.transform(X)
        
        return X
    
# Custom data/feature preprocessor
data_preprocessor = DataPreprocessor()


# scaler for numeric features
scaler = RobustScaler()

# one-hot encoder for categorical features
ohe = OneHotEncoder(drop='first', handle_unknown='ignore')


# Transformer to assign feature names if missing and handle data of different formats like dictionaries, lists or arrays for inference
class FeatureNamer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Handle Format 1: List of dictionaries
        if isinstance(X, list) and isinstance(X[0], dict):
            X = pd.DataFrame(X)

        # Handle Format 2: List of lists/arrays
        elif isinstance(X, list | np.ndarray) and isinstance(X[0], list | np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Handle Format 3 & 4: Dictionary of lists
        elif isinstance(X, dict):
            # Check if the dictionary keys match feature names
            if set(X.keys()) == set(self.feature_names):
                # If feature names are provided as keys, convert the dictionary directly
                X = pd.DataFrame(X)
            else:
                # dictionary with a list of lists under any key 
                for key in X:
                    if isinstance(X[key], list) and isinstance(X[key][0], list):
                        X = pd.DataFrame(X[key], columns=self.feature_names)
                        break

        # Handle DataFrames
        elif isinstance(X, pd.DataFrame):
            if X.columns.tolist() != self.feature_names:
                X.columns = self.feature_names
                
        else:
            # For any other format, assume it's a raw array/list of values
            X = pd.DataFrame(X, columns=self.feature_names)

        return X


# Wrap ColumnTransformer to access dynamic feature names from the custom transformer
class ColumnTransformerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.column_transformer = None

    def fit(self, X, y=None):
        self.preprocessor.fit(X)
        numeric_cols = self.preprocessor.numeric_cols
        categorical_cols = self.preprocessor.categorical_cols
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', scaler, numeric_cols),
                ('cat', ohe, categorical_cols)
            ],
            remainder = 'drop'
        )
        self.column_transformer.fit(X, y)
        return self

    def transform(self, X):
        return self.column_transformer.transform(X)

# Initialize the ColumnTransformerWrapper
column_transformer = ColumnTransformerWrapper(data_preprocessor)
