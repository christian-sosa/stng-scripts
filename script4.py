import warnings
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
from datetime import datetime as dt
import boto3
import json

s3resource = boto3.resource('s3')
s3 = boto3.client('s3')

warnings.filterwarnings("ignore")

def getFolder():
    HOY=dt.today()
    dia = HOY - timedelta(days=2)
    dia = dia.strftime('%A')
  
    dayOfWeek = ["Friday", "Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
    i=0
    for x in dayOfWeek:
        if dia == x:
            aux = i
        i += 1
        
    if aux == 0:
        folder = HOY - timedelta(days=2)
        folder = folder.strftime('%Y%m%d')
        folder = folder + '/'
    else:
        count = 2 + aux
        folder = HOY - timedelta(days=count)
        folder = folder.strftime('%Y%m%d')
        folder = folder #+ '/'
    return folder

#perfect
def df_loader():
    """
    Function implemented as DataFrame loader of *csv files corresponding to high power engines data, which are
    identified with their unique serial number (ESN), e.g., 66301108.csv.
    :param file_path: path to the folder containing all ESN Dataframes stored as *.csv
    :return: Sorted list containing *.csv file for each ESN.

    >> path = "/dataIncrement_20210924-20211002/"
    >> loaded_data = df_loader(file_path=path)
    """
    folders = []
    i = 0
    my_bucket = s3resource.Bucket('stnglambdaoutput')
    folder = getFolder()
    folder = 'Paso3/' #+ folder
    for object_summary in my_bucket.objects.filter(Prefix=folder):
        if (i == 0):
            i += 1
        else: 
            folders.append(object_summary.key)
    return folders

#perfect
def esn_list(data):
    esn_list_ = [str(data[i].split('/')[-1].split('.')[0]) for i in range(len(data))]
    return esn_list_


def df_inc_loader(esn,folder):
    pathforcsv = 's3://stnglambdaoutput/Paso3/'
    df = pd.read_csv(pathforcsv+str(esn)+".csv", sep=',', parse_dates=["devicetimestamp"], index_col="devicetimestamp")
    return df


def unit_adh(ad_h_df, esn, unit, site="Collahuasi"):

    if esn in ad_h_df.esn.unique():
        advance_h_unit = ad_h_df[(ad_h_df["Faena"] == site) & (ad_h_df["esn"] == esn) & (ad_h_df["Unidad"] == unit)]
        assert advance_h_unit.shape[0] != 0
        if advance_h_unit.shape[0] == 0:
            raise ValueError('engine mining site does not match in consolidado horometro dataset')
    else:
        raise ValueError('engine esn or site not in consolidado horometro dataset')

    return advance_h_unit

def synchronizer(atf_engine_resampled, ad_h_unit):

    if atf_engine_resampled.index[0] >= ad_h_unit.index[0] and atf_engine_resampled.index[-1] <= ad_h_unit.index[-1]:
        start = atf_engine_resampled.index[0]
        end = atf_engine_resampled.index[-1]
        atf_engine_resampled.loc[start:end, "engine_op_h"] = ad_h_unit.loc[start:end, "Horometro_Motor"]

    elif atf_engine_resampled.index[0] >= ad_h_unit.index[0] and atf_engine_resampled.index[-1] >= ad_h_unit.index[-1]:
        start = atf_engine_resampled.index[0]
        end = ad_h_unit.index[-1]
        atf_engine_resampled.loc[start:end, "engine_op_h"] = ad_h_unit.loc[start:end, "Horometro_Motor"]

    elif atf_engine_resampled.index[0] <= ad_h_unit.index[0] and atf_engine_resampled.index[-1] <= ad_h_unit.index[-1]:
        start = ad_h_unit.index[0]
        end = atf_engine_resampled.index[-1]
        atf_engine_resampled.loc[start:end, "engine_op_h"] = ad_h_unit.loc[start:end, "Horometro_Motor"]

    elif atf_engine_resampled.index[0] <= ad_h_unit.index[0] and atf_engine_resampled.index[-1] >= ad_h_unit.index[-1]:
        start = ad_h_unit.index[0]
        end = ad_h_unit.index[-1]
        atf_engine_resampled.loc[start:end, "engine_op_h"] = ad_h_unit.loc[start:end, "Horometro_Motor"]

    else:
        atf_engine_resampled.loc[ad_h_unit.index[0]:ad_h_unit.index[-1], "engine_op_h"] = \
            ad_h_unit.loc[ad_h_unit.index[0]:ad_h_unit.index[-1], "Horometro_Motor"]

    # assert atfEngineResampled.loc[start:end].shape[0]== avanceHUnit.loc[start:end,"Horometro_Motor"].shape[0]
    atf_engine_resampled["rul"] = atf_engine_resampled.index.map(lambda x: (x[-1] - x) / np.timedelta64(3600 * 24, 's'))
    atf_engine_resampled["esn"] = np.repeat(ad_h_unit.esn.unique(), atf_engine_resampled.shape[0])

    #atf_engine_resampled["Year"] = [str(x)[:4] for x in atf_engine_resampled["devicetimestamp"]]
    #atf_engine_resampled["Month"] = [str(x)[5:7] for x in atf_engine_resampled["devicetimestamp"]]
    #atf_engine_resampled["Day"] = [str(x)[8:10] for x in atf_engine_resampled["devicetimestamp"]]


    #for i, row in atf_engine_resampled.iterrows():
        


    return atf_engine_resampled

def op_hours_comp(df_, df_ad_h, unit, esn):

    idx = np.where(df_.engine_op_h.isnull())

    next_ = []
    prior = df_['engine_op_h'].iloc[idx[0][0] - 1]
    avg_adv_h = df_ad_h[(df_ad_h["Unidad"] == unit) & (df_ad_h["esn"] == esn)]["Avance"].mean().round()
    for i in range(len(idx[0])):
        next_.append(prior + avg_adv_h)
        prior = next_[i]

    df_['engine_op_h'].iloc[idx] = next_
    return df_

def weekly_increment_sync(esn_list_, ad_df, folder):
    units_df=pd.read_excel('s3://stnglambdainput/PedroCancino-DemoApi_A.xlsx')
    for i, esn in enumerate(esn_list_):
        esn = int(esn)
        unit_ = units_df[units_df["Placa"] == esn]["Unidad"].iloc[0]
        unit_ = int(unit_)
        df = df_inc_loader(esn=esn, folder=folder)
        ad_h_df_ = unit_adh(ad_h_df=ad_df, esn=esn, unit=unit_)
        df_ = synchronizer(atf_engine_resampled=df, ad_h_unit=ad_h_df_)
        df_ = op_hours_comp(df_, df_ad_h=ad_df, unit=unit_, esn=esn)
        nombreBucketPut ='s3://stnglambdaoutput/'
        df_.to_csv(path_or_buf=nombreBucketPut+folder+str(esn)+".csv", index_label='devicetimestamp')
        df_.to_parquet(nombreBucketPut+folder+str(esn)+".gzip.parquet",index='devicetimestamp', compression='gzip')


def lambda_handler(event, context):
    advanced_h = pd.read_csv(
        's3://stnglambdainput/consolidado_horometro_QSK78_.csv',
        delimiter=',',
        header=0,
        thousands=',',
        parse_dates=["Fecha"],
        dayfirst=True,
        index_col="Fecha"
    )

    nombreBucketDestino = 'stnglambdaoutput'
    folder = getFolder()
    folder = 'Paso4/' + folder + '/'
    s3.put_object(Bucket=nombreBucketDestino, Key=folder)

    data = df_loader()
    esn_list_in = esn_list(data=data)
    weekly_increment_sync(esn_list_=esn_list_in, ad_df=advanced_h, folder = folder)

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
