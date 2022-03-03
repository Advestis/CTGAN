import tensorflow as tf
import pandas as pd
import os
from ctgan import CTGANSynthesizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df_real_data = pd.read_csv('./data_set.csv', index_col=0)
discrete_columns = ['season', 'weekends', 'hour range']
ctgan = CTGANSynthesizer(epochs=10, cuda=True, verbose=True)

# first call
ctgan.fit(df_real_data, discrete_columns)

# second call
ctgan.fit(df_real_data, discrete_columns)
