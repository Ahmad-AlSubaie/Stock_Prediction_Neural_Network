import investpy
import numpy as np

def get_useable_data_lable_pair(start, end):
    df = investpy.get_currency_cross_historical_data(currency_cross='USD/EUR', from_date=start, to_date=end)
    
    del df['Currency']
    df = df.reset_index()
    del df['Date']
    df = df.rename_axis(None)
    
    data_x = np.array([])
    info_x = np.array([])
    temp = np.array([])
    for (i,j) in df.iterrows():
        if((i+1)%15 != 0 ):
            #print(df.iloc[[i]].to_numpy())
            temp = np.append(temp , df.iloc[[i]].to_numpy()[0])
        if((i+1)%15 == 0 ):
            if(np.array_equal(data_x, [])):
                data_x = temp
                info_x = df.iloc[[i]].to_numpy()[0]
            else:
                data_x = np.vstack([data_x, temp])
                info_x = np.vstack([info_x, df.iloc[[i]].to_numpy()[0]])
            temp = []
    return (data_x, info_x)
