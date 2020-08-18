import investpy
import numpy as np

def get_stock_data(start, end):
    df = investpy.get_currency_cross_historical_data(currency_cross='USD/EUR', from_date=start, to_date=end)
        
    del df['Currency']
    df = df.reset_index()
    del df['Date']
    df = df.rename_axis(None)

    data_x = np.array([])

    for (i,j) in df.iterrows():
        if(np.array_equal(data_x, [])):
            data_x = [df.iloc[[i]].to_numpy()[0]]
        else:
            data_x = np.vstack((data_x , df.iloc[[i]].to_numpy()[0]))
    return data_x
