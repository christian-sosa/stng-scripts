import json
import pandas as pd
from datetime import timedelta
from datetime import datetime as dt
import time
import os
import requests
from requests.adapters import HTTPAdapter
from requests.exceptions import ConnectionError
import boto3

import glob
#from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

s3 = boto3.client('s3')

datenow=dt.today()
REQUESTER='DDIAZ_DATADOWNLOAD'
HOY=dt.today().strftime('%Y%m%d')
DIRECTORY=''.join([REQUESTER,'_',HOY])
SAMPLES=None  #None, todo lo que haya, sino 1500 --- 1500 puntos en el periodo de descarga
PARAMETER_LIST='8,9,13,14,15,16,20,21,22,23,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,52,53,827,60,61,999'
ENGINE_STATE=''
RPMFROM=''
RPMTO=''
PEDALFROM=''
PEDALTO=''
POWERFROM=''
POWERTO=''
LOADFACTORFROM=''
LOADFACTORTO=''
##dias de ciclo
DIAS_CICLOS=1   #como maximo 14 dias
##credenciales
API_CLIENT = 'CA5kohqLSUMkA0CB74M9uULTM0Qu8552'
API_SECRET = 'f8rN9fLwrziJJHzaBzSqZHDRn67kJGRc5W-zABrBEIg07X85eg1ICGZRfPHKtrEP'


def parse_date_inicio(date=None):
    """
    Convierte la fecha en string en formato YYYY-MM-DD
    Si no se especifica el dia, retorna el dia anterior (para procesamiento diario)
    :param date: fecha a convertir
    :return: fecha en formato YYYY-MM-DD como string
    """
    final_date = (dt.now() - timedelta(days=2)).date()
    return final_date

def parse_date_fin(date=None):
    """
    Convierte la fecha en string en formato YYYY-MM-DD
    Si no se especifica el dia, retorna el dia anterior (para procesamiento diario)
    :param date: fecha a convertir
    :return: fecha en formato YYYY-MM-DD como string
    """
    final_date = (dt.now() - timedelta(days=2)).date()
    return final_date



##funcion gettoken
def getToken():
    print('Obteniendo el token ')
    ## hardcodea la url de la api
    url = 'https://bithaus.us.auth0.com/oauth/token'
    ##arma el payload con los datos de las credenciales
    payload = {"client_id":API_CLIENT,
            "client_secret":API_SECRET,
            "audience":"specto.cummins.cl/api","grant_type":"client_credentials"}
    ## arma el header
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    ## The maximum number of retries each connection should attempt. Note, this applies only to failed DNS lookups, socket connections and connection timeouts, never to requests where data has made it to the server. By default, Requests does not retry failed connections. If you need granular control over the conditions under which we retry a request, import urllib3’s Retry class and pass that instead.
    api_adapter = HTTPAdapter(max_retries=15)
    ##The Session object allows you to persist certain parameters across requests
    session = requests.Session()
    session.mount('https://bithaus.us.auth0.com', api_adapter)
    try:
        #r = requests.post(url, data=payload)#, headers=headers)
        ## arma el request POST a la url con el PAYLOAD formado anteriormente
        r = session.post(url, data=payload)
        ## IF para ver si fallo o no
        if r.status_code is 200:
            ##consigue el token y las cookies y las devuelve
            token=r.json()['access_token']
            cookies=str(r.cookies).split(',')[0].split('=')[1].split(' ')[0]
            #print('el token es: ', token)
            return token, cookies
        else:
            print('Hubo un problema')
            return None
    except ConnectionError as ce:
        print(ce)

#funcion descargaDia
def descargaDia(token,cookies, esn, minDate, maxDate):
    #if maxDate > dt(2021,2,15):
    #    maxDate='2021-01-31 00:00:00'
    
    ## The maximum number of retries each connection should attempt. Note, this applies only to failed DNS lookups, socket connections and connection timeouts, never to requests where data has made it to the server. By default, Requests does not retry failed connections. If you need granular control over the conditions under which we retry a request, import urllib3’s Retry class and pass that instead.
    ## arma el adapter y crea la session y monta OTRA url
    api_adapter = HTTPAdapter(max_retries=50)
    session = requests.Session()
    session.mount('https://specto.cummins.cl', api_adapter)

    ## arma la url con ESN 
    url = ''.join(['https://specto.cummins.cl/api/spectoParams/',str(esn)])
    
    ##formatea las fechas
    fechaMenor=''.join(['{"minDate":"',str(minDate)])
    fechaMayor=''.join(['","maxDate":"',str(maxDate).replace('00:00:00','23:59:59'),'"'])

    
    ## validaciones de formato
    if SAMPLES is None:
        samples=''
    else:
        samples=''.join([',"samples":',str(SAMPLES)])

    if PARAMETER_LIST is '':
        parameterList='null'
    else:
        parameterList=''.join(['"',PARAMETER_LIST,'"'])

    if ENGINE_STATE is '':
        engineState='null'
    else:
        engineState=''.join(['"',ENGINE_STATE,'"'])

    if RPMFROM is '':
        rpmFrom='null'
    else:
        rpmFrom=''.join(['"',RPMFROM,'"'])

    if RPMTO is '':
        rpmTo='null'
    else:
        rpmTo=''.join(['"',RPMTO,'"'])

    if PEDALFROM is '':
        pedalFrom='null'
    else:
        pedalFrom=''.join(['"',PEDALFROM,'"'])

    if PEDALTO is '':
        pedalTo='null'
    else:
        pedalTo=''.join(['"',PEDALTO,'"'])

    if POWERFROM is '':
        powerFrom='null'
    else:
        powerFrom=''.join(['"',POWERFROM,'"'])

    if POWERTO is '':
        powerTo='null'
    else:
        powerTo=''.join(['"',POWERTO,'"'])

    if LOADFACTORFROM is '':
        loadfactorFrom='null'
    else:
        loadfactorFrom=''.join(['"',LOADFACTORFROM,'"'])

    if LOADFACTORTO is '':
        loadfactorTo='null'
    else:
        loadfactorTo=''.join(['"',LOADFACTORTO,'"'])

    
    ## arma request con todos los datos validados anteriormente
    request = ''.join([fechaMenor, fechaMayor, samples, ',"parameterList":', parameterList, ',"engineState":',engineState , ',"rpmFrom":', rpmFrom,
                   ',"rpmTo":', rpmTo, ',"pedalFrom":',pedalFrom, ',"pedalTo":', pedalTo, ',"powerFrom":', powerFrom,
                   ',"powerTo":', powerTo, ',"loadFactorFrom":', loadfactorFrom, ',"loadFactorTo":', loadfactorTo,'}' ])


    ## con el token que recibe arma el bearer para la peticion
    bearer='Bearer '+token
    ## arma la variable cookies y deja vacio el payload
    cookie='PHPSESSID='+cookies
    payload={}
    ## arma el header con los datos anteriores
    headers = {
      'request': request,
      'Authorization': bearer,
      'Cookie': cookie
    }
    #recibe el response al hacer el get
    response = session.get(url, headers=headers, data=payload)
    print(request)
    ## valida que llego el status 200
    if response.status_code is 200:
        ## arma la variable text con la respuesta.text
        texto=response.text
        try:
            if ',' in  texto[-4:]:
                #print('tiene coma, modificando el final del json')
                texto=texto[:-2]+"]"
            data=json.dumps(texto)
            #df=pd.DataFrame(json.loads(data))
            #del df
        except:
            texto=response.text
            if (texto[-1] is ','):
                print('cambio de coma')
                texto=texto[:-2]+"]"
            #print(texto)
            data=json.dumps(texto)
            #print("Error en json decode, descarga data")
        return data
    else:
        try:
            print(response.text)
            print(response.status_code, response.json())
            print('Hubo un problema, descargarDia....')
        finally:
            data = pd.DataFrame()
            return data

##funcion checktime
def checkTime(tiempo):
    ## hardcodea y hace una cuenta de verificacion
    HORAS = 8
    MINUTOS = 60
    SECONDS = 80
    tiempoVerificacion = HORAS*MINUTOS*SECONDS
    ## saca hora actual
    tiempoActual = dt.now()
    ## si la hora actual - la variable que llego es mayor a la cuenta hardcodeada devuelve TRUE , si no FALSE
    if (tiempoActual - tiempo).seconds > tiempoVerificacion:
        return True
    else:
        return False

def guardarData(unidad, placa, df, dia,url):
    #nombreBucketDestino = 'stnglambdaoutput'
    keyName = unidad+'_'+placa+'_'+dia.replace(' 00:00:00','')+'.csv'
    keyPut = 's3://stnglambdaoutput/'+url+keyName
    
    df=df.copy()
    df.to_csv(keyPut, sep=';',index=False)

def lambda_handler(event, context):
    url = str(event['Url'])
    placa = str(event['Placa'])
    unidad = str(event['Unidad'])
    fechaAux = str(parse_date_inicio())
    fechaAux2 = str(parse_date_fin())
    fechaps = fechaAux + " 00:00:00"
    fechafin = fechaAux2 +  " 00:00:00"

    token, cookies = getToken()
     
    data=descargaDia(token, cookies, placa, fechaps, fechafin)

    if (data is not None):
        if (len (data) > 4):
            dfTemp=pd.read_json(json.loads(data))
            dfParam=dfTemp['paramvaluemap'].map(eval)
                
            dfParam=pd.json_normalize(dfParam)
            df=pd.concat([dfTemp,dfParam],axis=1).drop('paramvaluemap',axis=1)
            guardarData(unidad, placa ,  df, fechaps,url)
            del dfTemp
            del dfParam
            del data
            del df
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!'),
    }


