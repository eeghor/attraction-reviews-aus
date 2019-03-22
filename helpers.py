import pandas as pd
from scipy.stats import norm

def selector(df, req_dict):

    """
    returns a filtered data frame created from data frame df as specified by dictionary req_dict; 
    """

    if not isinstance(df, pd.DataFrame):
        raise Exception('argument df must be a pandas dataframe!')
    elif df.empty:
        raise Exception('dataframe you provided is empty!')

    actual_cols = set(df.columns)
    required_cols = set(req_dict)

    if not (required_cols <= actual_cols):

        cols_na = ', '.join(required_cols - actual_cols)

        raise Exception(f'column(s) {cols_na} you\'re asking for are not available!')

    out = df

    for col in required_cols:
        
        out = out[out[col].astype(str).apply(lambda x: req_dict[col] in x)]
          
        if out.empty:
            break
    
    return out   

def normcdf(ser):

    if not isinstance(ser, pd.Series):

        raise Exception('argument ser must be a pandas series!')

    return norm.cdf(ser, ser.mean(), ser.std()) 

