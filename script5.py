import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import timedelta
from datetime import datetime as dt
import boto3

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from weeklyIncrement_ import concat_list, weekly_increment
from weeklyDataIncrementSync import df_loader, esn_list, weekly_increment_sync

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

tfd = tfp.distributions
tfkl = tf.keras.layers
tfpl = tfp.layers

mpl.rcParams["figure.figsize"] = (11,7)
sns.set_theme(context='notebook',style="whitegrid")
plt.style.use('bmh')

s3resource = boto3.resource('s3')
s3 = boto3.client('s3')

def file_loader():
    ## recibe un esn y un path
    ## y devuelve una lista de los pathnames que coincidan con el path que armo con esos datos
    folders = []
    my_bucket = s3resource.Bucket('stnglambdaoutput')
    folder = getFolder()
    folder = 'Paso4/' + folder
    i = 0
    for object_summary in my_bucket.objects.filter(Prefix=folder):
        if i == 0:
            i += 0
        else:
            folders.append(object_summary.key)
    print(folders)
    return folders


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

# ok
def data_frame(data):
    kwargs = {'delimiter': ',',
              "thousands": ',',
              "header": 0,
              'na_values': np.nan,
              'parse_dates': ["devicetimestamp"],
              "index_col": "devicetimestamp"}
    dff = pd.read_csv(data[0], **kwargs)
    dff.interpolate(inplace=True)
    for i in range(1, len(data)):
        df_ = pd.read_csv(data[i], **kwargs)
        df_.interpolate(inplace=True)
        dff = dff.append(df_)

    return dff


sz: int


def kalman_filter(data_, xhat, P, xhatminus, Pminus, K, R, sz: int):
    # initial parameters
    z = data_  # observations
    Q = 1e-5  # process variance

    # Initial guess
    xhat[0] = data_[0]
    #P[0] = 1.0

    for k in range(1, sz):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q
        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (z[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat
   

def kalman_viz(data,col):
    plt.plot(data, 'k+', label='noisy measurement')
    plt.plot(xhat, 'b-', label='a posteri estimate')
    plt.legend()
    plt.title(col)
    plt.xlabel('Iteration')
    plt.ylabel('Measurement')
    plt.show()


def classCounter(df_f):
    tbo_count = 0
    pl_count = 0
    cs_count = 0
    ind_count = 0
    for esn in df_f.index:
        if df_f.loc[esn,"TBO"] >= 0.5:
            tbo_count += 1
        elif df_f.loc[esn, "Seized_PL"] >= 0.5:
            pl_count += 1
        elif df_f.loc[esn, "Seized_CkL"] >= 0.5:
            cs_count +=1
        else:
            ind_count += 1
    print(f"TBOs: {tbo_count}, Vent: {pl_count}, Cronn rod seized: {cs_count}, IND: {ind_count}")
    
    plt.bar("TBO", height=tbo_count, alpha=.7)
    plt.bar("Vent", height=pl_count, alpha=.7)
    plt.bar("Cronn rod seized", height=cs_count, alpha=.7, edgecolor='b')
    plt.bar("Indeterminated", height=ind_count, alpha=.7, edgecolor='b')
    plt.show()
    
#ok
def classID(df_f):        
        
    for esn in df_f.index:
        df_f.loc[esn,'op_h'] = dff[dff.esn==esn].loc[:,'engine_op_h'].max()
        
        if df_f.loc[esn,"TBO"] >= 0.5:
            df_f.loc[esn,"Failure_mode"] = "TBO/SUSP"
        elif df_f.loc[esn, "Seized_PL"] >= 0.5:
            df_f.loc[esn,"Failure_mode"] = "VENTILADO"
        elif df_f.loc[esn, "Seized_CkL"] >= 0.5:
            df_f.loc[esn,"Failure_mode"] = "FUNDIDO"
        else:
            df_f.loc[esn,"Failure_mode"] = "IND"
            
            
    return df_f


def create_dataset(X, y, time_step=1):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        v = X.iloc[i:(i + time_step)].values
        Xs.append(v)
        ys.append(y.iloc[(i + time_step)])

    return np.array(Xs), np.array(ys)


def nll(ytrue, ypred):
    return -ypred.log_prob(ytrue)


divergence_fn = lambda q,p,_ : tfd.kl_divergence(q,p)/19930


@tf.function
def nll(label_true, label_pred):
    return -label_pred.log_prob(label_true)


def class_model():
    modelj = tf.keras.Sequential()
    modelj.add(tfkl.InputLayer(input_shape=(30, 15)))

    modelj.add(tfkl.Conv1D(filters=32, kernel_size=58,
                           padding='causal', strides=1, dilation_rate=1
                           ))
    modelj.add(tfkl.BatchNormalization())
    modelj.add(tfkl.ReLU())
    modelj.add(
        tfkl.MaxPool1D())
    modelj.add(
        tfpl.Convolution1DFlipout(
            filters=8,
            kernel_size=2,
            padding='SAME',
            activation=tf.nn.sigmoid,
            strides=1,
            # dilation_rate=1,
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence_fn,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence_fn
        ))
    modelj.add(
        tfkl.MaxPool1D())
    modelj.add(
        tfkl.Flatten())

    modelj.add(tfkl.BatchNormalization())

    modelj.add(tfpl.DenseFlipout(
        units=128,
        activation=tf.nn.sigmoid,  # "linear",
        kernel_divergence_fn=divergence_fn,
        bias_divergence_fn=divergence_fn,
        bias_prior_fn=tfpl.default_multivariate_normal_fn,
        bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False))
    )

    modelj.add(
        tfpl.DenseFlipout(
            units=tfpl.OneHotCategorical.params_size(3), activation=None,
            kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            kernel_divergence_fn=divergence_fn,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(is_singular=False),
            bias_divergence_fn=divergence_fn
        ))
    modelj.add(
        tfpl.OneHotCategorical(3)
    )

    modelj.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01, momentum=0.8),
        loss=nll,
        metrics=['accuracy'])

    weightsj = "/Users/danieldiazalmeida/Documents/TSModel_KomatsuCummins/modele_weights_30_accj_RT_e.h5"
    modelj.load_weights(weightsj)

    return modelj

#ok
def classifier(data_):
    a = []
    model = class_model()

    assert type(data_) == list, "data argument is not a list. It must be a list."

    for i in range(len(data_)):
        kwargs = {'delimiter': ',',
                  "thousands": ',',
                  "header": 0,
                  'na_values': np.nan,
                  'parse_dates': ["devicetimestamp"],
                  "index_col": "devicetimestamp"}
        df = pd.read_csv(data_[i], **kwargs)
        df = df.loc[:].apply(lambda col: np.where(col.isnull(), np.mean(col), col))
        df.interpolate(inplace=True)
        df = df.applymap(lambda x: 0 if x < 0 else x)
        df.interpolate(inplace=True)

        esn_df = df['esn']
        df = df.drop(['esn'], axis=1)

        if df.shape[0] > 30.0:

            data = {}
            for col in df.columns[:-2]:
                sz: int
                sz = len(list(df.index))
                # Allocate space for arrays
                xhat = np.zeros(sz)  # a posteri estimate of x
                P = np.zeros(sz)  # a posteri error estimate
                P[0] = 1.0
                xhatminus = np.zeros(sz)  # a priori estimate of x
                Pminus = np.zeros(sz)  # a priori error estimate
                K = np.zeros(sz)  # gain or blending factor
                R = 1.0 ** 3  # estimate of measurement variance, chanf¡ge to see effects
                xhat = kalman_filter(df[col].values, xhat, P, xhatminus, Pminus, K, R, sz)
                # kalman_viz(df[col].values, col)
                data[col] = xhat

            df_ = pd.DataFrame(data, index=df.index[0:], columns=df.columns[:-2])
            df_["op_h"] = df.engine_op_h
            df_["rul"] = df.rul
            df_["esn"] = esn_df.values[0:]

            df_ = df_.set_index(['esn', df.index])

            scaler = StandardScaler()
            x_cuop_std = pd.DataFrame(scaler.fit_transform(df_), columns=df_.columns, index=df_.index, dtype='float16')

            tf.random.set_seed(1234)

            for esn in list(esn_df.unique()):
                x_cuop_array, y_true_array = create_dataset(x_cuop_std.loc[esn], x_cuop_std.loc[esn, "egt"], time_step=30)
                a.append(np.sum(model.predict(x_cuop_array), axis=0) / x_cuop_array.shape[0])
                continue

    df_f = pd.DataFrame(np.asarray(a), index=dff.esn.unique(),
                        columns=["TBO", "Seized_PL", "Seized_CkL"])
    return df_f

#ok
def hours_aggregation(df_f, dff):
    for esn in df_f.index:
        df_f.loc[esn, 'op_h'] = dff[dff.esn == esn].loc[:, 'engine_op_h'].max()

    return df_f



if __name__ == "__main__":



    ## ----- Weekly Classification -----
    folderDate = getFolder()
    
    path_inc_sync = 's3://stnglambdaoutput/Paso4/' + folderDate  
    data1 = file_loader()
    data1.sort()

    dff = data_frame(data1)
    df_f = classifier(data_=data1)
    #classCounter(df_f)
    df_f = classID(df_f)
    df_f = hours_aggregation(df_f,  dff)

    pathForReport = 's3://stnglambdaoutput/Paso5/'+folderDate

    df_f.to_csv(pathForReport + ".csv", index_label="ESN")
    t2 = dt.datetime.now()

    print("-----------------------------")
    print(os.sys.version)
    print(f"numpy version: {np.__version__}")
    print(f"pandas version: {pd.__version__}")
    print(f"Tensorflow version: {tf.__version__}")
    print(f"Tensorflow-Probability version {tfp.__version__}")
    print("-----------------------------")
