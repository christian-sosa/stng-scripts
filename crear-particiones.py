import json
import awswrangler as wr
import boto3
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt

s3 = boto3.client('s3')

def lambda_handler(event, context):
    bucketName = "s3://" + event['Records'][0]['s3']['bucket']['name'] +"/"
    csvPath = event['Records'][0]['s3']['object']['key']
    
    df = pd.read_csv(bucketName+csvPath,delimiter= ',')
    
    wr.s3.to_parquet(df = df, path = "s3://stng-sm/dataset", dataset = True, partition_cols = ['Date'], compression = "gzip")
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
