import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
import json
import boto3

s3resource = boto3.resource('s3')
s3 = boto3.client('s3')

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
        folder = folder + '/'
    return folder

def df_loaderPath1():
    folders = []
    i = 0
    my_bucket = s3resource.Bucket('stnglambdaoutput')
    folder = getFolder()
    folder = 'Paso3/Historico/'
    for object_summary in my_bucket.objects.filter(Prefix=folder):
        if (i == 0):
            i += 1
        else: 
            folders.append(object_summary.key)
    return folders

def df_loaderPath2():
    folders = []
    i = 0
    my_bucket = s3resource.Bucket('stnglambdaoutput')
    folder = getFolder()
    folder = 'Paso2/' + folder
    for object_summary in my_bucket.objects.filter(Prefix=folder):
        if (i == 0):
            i += 1
        else: 
            folders.append(object_summary.key)
    return folders

def drop_cols(df, to_drop_list):
    to_drop = [col for col in to_drop_list if col in df.columns]
    df_ = df.drop(to_drop, axis=1)
    # print(df_.shape)
    return df_


def concat_list():
    #historico
    data1 = df_loaderPath1()
    #7 dias antes
    data2 = df_loaderPath2()

    prior_increment_list = [int(data1[i].split('/')[-1].split('.')[0]) for i in range(len(data1))]
    current_increment_list = [int(data2[i].split('/')[-1].split('.')[0]) for i in range(len(data2))]

    return current_increment_list, prior_increment_list


def df_prep(esn, print_info=False):
    esn = str(esn)
    path = 's3://stnglambdaoutput/Paso3/Historico/' + esn
    kwargs = {'delimiter': ',',
              "thousands": ',',
              "header": 0,
              'na_values': np.nan,
              'parse_dates': ["devicetimestamp"],
              "index_col": "devicetimestamp"}
    df = pd.read_csv(path + ".csv", **kwargs)

    if print_info is True: print(df.info())

    return df

def df_inc_prep(esn, print_info=False):
    esn = str(esn)
    date = getFolder()
    path = 's3://stnglambdaoutput/Paso2/'+ date + esn
    kwargs = {'delimiter': ';',
              "thousands": ',',
              "header": 0, 'na_values': np.nan,
              'parse_dates': ["devicetimestamp"],
              "index_col": "devicetimestamp"}
    df_inc = pd.read_csv(path + ".csv", **kwargs)
    #df_inc = df_inc.loc[:].apply(lambda col: np.where(col.isnull(), np.mean(col), col))
    df_inc["egt"] = df_inc[egt_listinc].mean(axis=1)
    df_inc["imt"] = df_inc[imt_listinc].mean(axis=1)
    df_inc["imp"] = df_inc[imp_listinc].mean(axis=1)

    df_inc = drop_cols(df_inc, egt_listinc)
    df_inc = drop_cols(df_inc, imp_listinc)
    df_inc = drop_cols(df_inc, imt_listinc)
    df_inc.drop(["esn"], axis=1, inplace=True)
    

    if print_info is True: print(df_inc.info())

    return df_inc

def df_concat(df, df_inc):
    print(df_inc.columns)
    print(df.columns)
    df.columns = df_inc.columns
    df_concat_ = df.append(df_inc)
    assert df_concat_.shape[0] == df.shape[0] + df_inc.shape[0]

    return df_concat_

def weekly_increment(current_list, prior_list):
    date = getFolder()
    folder_PATH = 's3://stnglambdaoutput/Paso3/' + date

    ## itera la lista de esn de 7 dias
    for i,esn in enumerate(prior_list):
        ## si esta en el historico
        if esn in current_list:
            print(i,esn)
            df = df_prep(esn=esn)
            df_inc = df_inc_prep(esn=esn)
            df_concat_ = df_concat(df, df_inc)
            df_concat_.to_csv(folder_PATH+str(esn)+".csv", index_label='devicetimestamp')
        else:
            print(f"{i} {esn} no in current_list")
            df_inc = pd.read_csv('s3://stnglambdaoutput/Paso3/Historico/' +str(esn)+'.csv', 
                                 parse_dates=["devicetimestamp"], 
                                 index_col="devicetimestamp"
                                 ,delimiter= ',')
            df_inc.to_csv(folder_PATH+ str(esn)+".csv", index_label='devicetimestamp')

egt_listinc = ['EGT-01', 'EGT-02', 'EGT-03', 'EGT-04', 'EGT-05', 'EGT-06', 'EGT-07',
               'EGT-08', 'EGT-09', 'EGT-10', 'EGT-11', 'EGT-12', 'EGT-13', 'EGT-14',
               'EGT-15', 'EGT-16', 'EGT-17', 'EGT-18']
imp_listinc = ['IMP-LB', 'IMP-RB', 'IMP-RB (MCRS)']
imt_listinc = ['IMT-LBF', 'IMT-LBR', 'IMT-RBF', 'IMT-RBR', 'IMT-LBM', 'IMT-RBM']

def lambda_handler(event, context):
    current_increment_list, prior_increment_list = concat_list()
    print(current_increment_list)
    print(prior_increment_list)

    weekly_increment(current_list=current_increment_list, prior_list=prior_increment_list)
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }



