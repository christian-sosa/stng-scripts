## importa todo lo requerido
import json
import dask
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
import boto3
from datetime import datetime as dt
from dask.delayed import delayed
from dask.diagnostics import ProgressBar

s3resource = boto3.resource('s3')
s3 = boto3.client('s3')

def getFolder():
    HOY=dt.today()
    dia = (HOY - timedelta(days=3)).strftime('%A')
  
    dayOfWeek = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    i=0
    for x in dayOfWeek:
        if x == dia:
            aux = i
        i += 1
        
    count = 4 - aux
    folder = (HOY + timedelta(days=count)).strftime('%Y%m%d') + '/'
    return folder

##OK
def esn_list():
    folders = []
    my_bucket = s3resource.Bucket('stnglambdaoutput')
    folder = getFolder()
    folder = 'Paso1/' + folder
    for object_summary in my_bucket.objects.filter(Prefix=folder):
        if (object_summary.key == folder):
            continue
        else: 
            folders.append(object_summary.key)

    folders.sort()
    fileDF = pd.DataFrame(
        np.array(
            [folders[i].split('_') for i in range(len(folders))]
        )
    )

    #fileDF.columns = ["file_index","UNIT","ESN","FILE_EXTENSION"]
    fileDF.columns = ["UNIT","ESN","FILE_EXTENSION"]
    ## arma una lista con los campos unicos de fileDF columna ESN
    esns = list(fileDF["ESN"].unique())
    ## devuelve data y folders
    return esns#, esns


def file_loader(esn):
    ## recibe un esn y un path
    ## y devuelve una lista de los pathnames que coincidan con el path que armo con esos datos
    folders = []
    my_bucket = s3resource.Bucket('stnglambdaoutput')
    folder = getFolder()
    folder = 'Paso1/' + folder
    for object_summary in my_bucket.objects.filter(Prefix=folder):
        aux = object_summary.key
        x = aux.find(esn)
        if x != -1:
            folders.append(aux)

    return folders


#@dask.delayed
## no se esta usando
def down_sampler(df): #-> DataFrame
    ## Convenience method for frequency conversion and resampling of time series. The object must have a datetime-like index (DatetimeIndex, PeriodIndex, or TimedeltaIndex), or the caller must pass the label of a datetime-like series/index to the on/level keyword parameter.
    df_ = df.resample('D').mean()
    return df_.copy()

##OK
def data_loader(files):
    ## arma los parametros que enviara
    kwargs = {"delimiter":';', 
              "header":0, "decimal":'.', 
              "keep_default_na":True, 
              #"usecols":columns, 
              "parse_dates":["devicetimestamp"], 
              "index_col":"devicetimestamp"}
    ## con el dato que recibe FILES lee el csv y arma un df
    pathforcsv = 's3://stnglambdaoutput/'

    df = pd.read_csv(pathforcsv+files[0], **kwargs)
    ## lo ordena
    df.sort_index(inplace=True)
    df = df.resample("D").mean()
    ## itera file
    for file in files[1:]:
        ## arma otro df
        #df_ = pd.read_csv(pathforcsv+folder+file, **kwargs)
        df_ = pd.read_csv(pathforcsv+file, **kwargs)
        df_.sort_index(inplace=True)
        df_ = df_.resample('D').mean()
        df = pd.concat([df, df_])
        
    return df.copy()
##no se esta usando
def paramsNN(df):
    ##  lee el csv
    paramNN = pd.read_csv("parametroNN.csv", header=0, sep=',',encoding= 'unicode_escape')
    params = {}
    ## crea un objeto vacio 
    ## itera param
    for name, id_ in zip(paramNN["nombre"],paramNN["id_parametro_specto"]):
        params[str(id_)] = name
        
    ## al df que recibe le cambia el nombre de las columnas
    df.rename(columns=params, inplace=True)
    
    
    return df

##OK
def cols_adquirer():
    FILE_REQUEST = 's3://stnglambdainput/parametroNN.csv'
    paramNN=pd.read_csv(FILE_REQUEST, header=0, sep=',',encoding= 'unicode_escape')
    ## lee el csv
    params = {str(id_): name for name, id_ in zip(paramNN["nombre"],paramNN["id_parametro_specto"])}
    cols = ['esn','rpm']
    cols.extend(list(params.values()))
    return cols, params


egt_listinc = ['EGT-01', 'EGT-02', 'EGT-03', 'EGT-04', 'EGT-05', 'EGT-06', 'EGT-07',
               'EGT-08', 'EGT-09', 'EGT-10', 'EGT-11', 'EGT-12', 'EGT-13', 'EGT-14',
               'EGT-15', 'EGT-16', 'EGT-17', 'EGT-18','IMP-LB', 'IMP-RB', 'IMP-RB (MCRS)',
               'IMT-LBF', 'IMT-LBR', 'IMT-RBF', 'IMT-RBR', 'IMT-LBM', 'IMT-RBM']


def useful_stuff(data_loader, files ,params, cols, folder=None):
    df = delayed(data_loader)(files)
    with ProgressBar():
        df_ = df.compute()
        
    df_.sort_index(inplace=True)
    df_ = df_.apply(lambda col: col.mask(col.isna(),np.mean(col)), axis=0)
    for col in params.keys():
        if col in df_.columns:
            df_.rename(columns={col:params[col]}, inplace=True)
        else:
            df_.loc[:,params[col]] = np.nan
    
    df_ = df_[cols]


    #seteo path
    nombreBucketDestino = 'stnglambdaoutput'
    nombreBucketPut ='s3://stnglambdaoutput/'

    name = getFolder()
    folder2 = folder + name

    s3.put_object(Bucket=nombreBucketDestino, Key=folder2)
    df_.to_csv(path_or_buf=nombreBucketPut+folder2+str(int(df_.esn.unique()))+".csv",sep=';')

##OK
def computing(esn_list,params, cols, folder=None):
    for esn in esn_list:
        files = file_loader(esn)
        useful_stuff(data_loader, files=files, params=params, cols=cols, folder=folder) 
    


path = "~/Documents/DataFramesCompacted/training/"
##no se esta usando
def df_loader(esn, path=path): 
    """
    esn      =  engine serial number
    path     =  complete path to the folder containing engine data *.csv which delimiter is ';'
    
    Returns
    -----------------
        Pandas DataFrame with device's tinestamp as index
    
    """
    
    kwargs = {
        "sep":';',
        "header":0,
        "parse_dates":["devicetimestamp"],
        "index_col":"devicetimestamp"
    }
    df = pd.read_csv(path+str(esn)+".csv", **kwargs)
    
    return df

def lambda_handler(event, context):
    cols, params = cols_adquirer()
    data_esns1 = esn_list() 
    folder = 'Paso2/'

    computing(esn_list=data_esns1,params=params, cols=cols, folder=folder)

    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
