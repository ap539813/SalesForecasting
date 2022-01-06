import altair as alt
import plotly.express as px
import datetime
import copy
import numpy as np
import pandas as pd
import streamlit as st
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler


# Now we will define function to get train the model and return the model summary and performance

def Model_result():
  # Create the input and output dat
  st.markdown(
      "### Retrain the model on an updated dataset"
  )
  st.markdown(
      "  \n   "
  )
  files = os.listdir('Data/')
  files = [file for file in files if file[-3:] == 'csv']
  filename = st.selectbox('Select the data',[''] + files)
  click = st.button('Train Model')
  if click:
      if filename != '':
          data = generate_data(filename)
          data_update = pd.DataFrame()
          data_update['Sales'] = data[['2017', '2018', '2019', '2020', '2021']].sum(axis = 1)
          data_update.dropna(inplace=True)
          data_update.columns = ['Sales']
          data_update = data_update.groupby('Dates').sum()
          data_update['change_in_amonth'] = (data_update['Sales'] - data_update['Sales'].shift(1))
          for i in range(24):
            name_of_col = 'attibute_' + str(i)
            data_update[name_of_col] = data_update['change_in_amonth'].shift(i+1)

          data_update.dropna(inplace = True)
          data_update['pred'] = 0
          data_update['pred'][0] = data_update['Sales'].max()
          data_update['pred'][1] = data_update['Sales'].min()
          data_update = data_update[['Sales'] + list(data_update.columns[2:])]
          scaler = MinMaxScaler(feature_range=(-1, 1))
          scaler.fit(data_update)
          final_data_scaled = scaler.transform(data_update)
          train_set, test_set = final_data_scaled[:-20], final_data_scaled[-20:]
          X_train, y_train = train_set[:, 1:-1], train_set[:, 0:1]
          X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

          X_test, y_test = test_set[:, 1:-1], test_set[:, 0:1]
          X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

          model = Sequential()
          model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
          model.add(Dense(4))
          model.add(Dense(8))
          model.add(Dense(1))
          model.compile(loss='mean_squared_error', optimizer='adam')
          model.fit(X_train, y_train, epochs = 100, batch_size=1, verbose=1, shuffle=False)
          history = model.history.history
          model.save("model.hdf5")
          fig = px.line(history['loss'])
          st.plotly_chart(fig, use_container_width=True)
          score = model.evaluate(X_test, y_test, verbose = 0, batch_size = 1)
          st.write('Test loss:', score)
          pred = model.predict(final_data_scaled[:, 1:-1].reshape((77, 1, 24)),batch_size=1)
          final_data_scaled[:, -1] = pred.reshape((77, ))
          data_back = scaler.inverse_transform(final_data_scaled)
          data_back = pd.DataFrame(data_back, index = data_update.index, columns = data_update.columns)
          fig = make_subplots(rows=1, cols=1)
          fig.add_trace(
              go.Line(x = data_back.index, y = data_back['Sales'].values),
              row=1, col=1
          )

          fig.add_trace(
              go.Line(x = data_back.index, y = data_back['pred'].values),
              row=1, col=1
          )

          fig.update_layout(title_text="Real and Processed data side by side")
          st.plotly_chart(fig, use_container_width=True)

def show_data():
    st.markdown(
        "### Have a look at your data"
    )
    st.markdown(
        "  \n   "
    )
    files = os.listdir('Data/')
    files = [file for file in files if file[-3:] == 'csv']
    filename = st.selectbox('Select the data',[''] + files)
    if filename != '':
        data = generate_data(filename)
        st.markdown("Original Data")
        st.write(data)

        data_update = pd.DataFrame()
        data_update['Sales'] = data[['2017', '2018', '2019', '2020', '2021']].sum(axis = 1)
        data_update.dropna(inplace=True)
        data_update.columns = ['Sales']
        data_update = data_update.groupby('Dates').sum()
        data_update['change_in_amonth'] = (data_update['Sales'] - data_update['Sales'].shift(1))
        for i in range(24):
          name_of_col = 'attibute_' + str(i)
          data_update[name_of_col] = data_update['change_in_amonth'].shift(i+2)

        data_update.dropna(inplace = True)
        data_update['pred'] = 0
        data_update['pred'][0] = data_update['Sales'].max()
        data_update['pred'][1] = data_update['Sales'].min()
        st.markdown('Processed Data')
        st.write(data_update)

        fig = make_subplots(rows=2, cols=1)
        for col in ['2017', '2018', '2019', '2020', '2021']:
            fig.add_trace(
                go.Line(x = data.Dates.values, y = data[col].values),
                row=1, col=1
            )

        fig.add_trace(
            go.Line(x = data_update.index, y = data_update['Sales'].values),
            row=2, col=1
        )

        fig.update_layout(title_text="Real and Processed data side by side")
        st.plotly_chart(fig, use_container_width=True)

def prediction():
    model = keras.models.load_model('model.hdf5')
    default = datetime.date.today() + datetime.timedelta(days=1)
    end_date = st.date_input('End date', default)
    data = generate_data('Data_Info.csv')
    data_update = pd.DataFrame()
    data_update['Sales'] = data[['2017', '2018', '2019', '2020', '2021']].sum(axis = 1)
    data_update.dropna(inplace=True)
    data_update.columns = ['Sales']

    data_update = data_update.groupby('Dates').sum()
    last_date = data_update.index.max().date()
    data_update.index = [d.date() for d in data_update.index]
    data_update.sort_index(inplace = True)
    idx = [last_date + datetime.timedelta(days=i) for i in range(1, (end_date - last_date).days)]
    for i in range(len(idx)):
        if idx[i].day < 16:
            idx[i] = datetime.date(idx[i].year, idx[i].month, 1)
        else:
            idx[i] = datetime.date(idx[i].year, idx[i].month, 16)

    idx = sorted(list(set(idx)))
    sale = [0 for i in idx]
    temp_days = pd.DataFrame(sale, index = idx, columns = ['Sales'])
    data_update = pd.concat([data_update, temp_days])
    data_update['change_in_amonth'] = (data_update['Sales'] - data_update['Sales'].shift(1))

    for i in range(24):
      name_of_col = 'attibute_' + str(i)
      data_update[name_of_col] = data_update['change_in_amonth'].shift(i+1)

    data_update.dropna(inplace = True)
    data_update['pred'] = 0
    data_update['pred'][0] = data_update['Sales'].max()
    data_update['pred'][1] = data_update['Sales'].min()
    data_update = data_update[['Sales'] + list(data_update.columns[2:])]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_update)
    matrix_data = data_update.values
    for d in range(list(data_update.index).index(last_date) + 1, data_update.shape[0]):
        final_data_scaled = scaler.transform(matrix_data[d-1:d+1, :])
        pred = model.predict(final_data_scaled[1, 1:-1].reshape((1, 1, 24)), batch_size=1)
        final_data_scaled[1, -1] = pred.reshape((1, ))
        final_data_scaled[1, 0] = pred.reshape((1, ))
        final_data_scaled = scaler.inverse_transform(final_data_scaled)
        try:
            matrix_data[d-1:d+1, :] = final_data_scaled
            temp = final_data_scaled[1, 0] - final_data_scaled[0, 0]
            matrix_data[d+1, 1] = temp
            matrix_data[d+1, 2:-1] = matrix_data[d, 1:-2]
        except:
            pass

    # data_back = scaler.inverse_transform(final_data_scaled)
    data_back = pd.DataFrame(matrix_data, index = data_update.index, columns = data_update.columns).loc[idx]
    st.write(data_back[['Sales']])
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Line(x = data_back.index, y = data_back['Sales'].values),
        row=1, col=1
    )

    fig.update_layout(width = 1600, height = 500, title_text="Real and Processed data side by side")
    st.plotly_chart(fig, use_container_width=True)



alt.renderers.set_embed_options(scaleFactor=2)


## Basic setup and app layout
st.set_page_config(layout="wide")  # this needs to be the first Streamlit command called
st.title("Sales forecasting")

st.sidebar.title("Control Panel")
left_col, middle_col, right_col = st.columns(3)
palette = sns.color_palette("bright")

tick_size = 12
axis_title_size = 16


## Simulate data and the distribution domain
@st.cache
def generate_data(filename):
    data = pd.read_csv(filename, index_col = 0)
    data = data[:-1]
    data.columns = ['Store', 'Type', 'Dates', '2017', '2018', '2019', '2020', '2021']
    data['Dates'] = [item[:-2]+'01' if int(item[-2:]) <= 15 else item[:-2]+'16' for item in data['Dates']]
    data['Dates'] = pd.to_datetime(data['Dates'], dayfirst = True, format = "%Y-%m-%d")
    # data.set_index('Dates', inplace = True)
    data.index = data['Dates']
    data.sort_index(inplace = True)
    return data


# data = generate_data()
mode = st.sidebar.radio("Select Section: ", ('Retrain the model',
'Look at Data',
'Prediction'))
if mode == 'Look at Data':
    show_data()
elif mode == 'Retrain the model':
    Model_result()
elif mode == 'Prediction':
    prediction()
