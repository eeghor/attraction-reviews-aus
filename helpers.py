import pandas as pd
from scipy.stats import norm

def selector(df, req_dict):

    """
    returns a pandas data frame produces by filtering df according to the requirements described
    in dictionary req_dict; for example, if req_dict = {'age': '20-24', 'gender': 'm'} the resulting data frame 
    will only contain males aged 20-24;

    * if the filtered data frame becomes empty, the function returns this empty data frame without issuing any warnings or
      exceptions
    """

    if not isinstance(df, pd.DataFrame):
        raise Exception('argument df must be a pandas dataframe!')
    elif df.empty:
        raise Exception('dataframe you provided is empty!')

    actual_cols = set(df.columns)
    required_cols = set(req_dict)

    out = df
    
    if not (required_cols <= actual_cols):

        cols_na = ', '.join(required_cols - actual_cols)

        raise Exception(f'column(s) {cols_na} you\'re asking for are not available!')

    for col in required_cols:
        
        out = out[out[col].astype(str).apply(lambda x: (req_dict[col] in x) if ('all' not in req_dict[col]) else True)]
          
        if out.empty:
            break
    
    return out   

def normcdf(ser):

    if not isinstance(ser, pd.Series):

        raise Exception('argument ser must be a pandas series!')

    return norm.cdf(ser, ser.mean(), ser.std()) 

