import json
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
import time
import boto3

invokeLan = boto3.client("lambda",region_name = "sa-east-1")
s3 = boto3.client('s3')

def getFolder():
    nombreBucketDestino = 'stnglambdaoutput'

    HOY=dt.today()
    folder = HOY - timedelta(days=2)
    folder = folder.strftime('%Y%m%d')
    folder = 'Paso1/' +folder + '/'
    
    dia = HOY - timedelta(days=2)
    dia = dia.strftime('%A')
  
    dayOfWeek = ["Saturday", "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    i=0
    for x in dayOfWeek:
        if x == dia:
            aux = i
        i += 1
        
    if aux == 0:
        s3.put_object(Bucket=nombreBucketDestino, Key=folder)
    else:
        count = 2 + aux
        folder = HOY - timedelta(days=count)
        folder = folder.strftime('%Y%m%d')
        folder = 'Paso1/' +folder + '/'
    return folder

def lambda_handler(event, context):
    FILE_REQUEST = 's3://stnglambdainput/PedroCancino-DemoApi_A.xlsx'

    folder = getFolder()
    df=pd.read_excel(FILE_REQUEST)
    dfDescarga=df.copy()

    for i, row in dfDescarga.iterrows():
        time.sleep(3)
        payload = {"Placa": row['Placa'],"Unidad": row['Unidad'],"Url": folder }
        invokeLan.invoke(FunctionName='DescargaCsvPlaca', InvocationType='Event', Payload=json.dumps(payload))

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
