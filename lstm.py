#Afaf Taik 
#GEI723
#Univariate LSTM

#from keras import backend as K
#from keras.engine.topology import Layer
from keras.layers import RNN, Activation,LSTM
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
import keras

import numpy
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

##########################################################################################
#Parameters
name = 'vanilla lstm'
n_lag = 5
n_seq = 3 
filename='house_by_minute_1_sum.csv'
verbose, epochs, batch_size = 2, 20, 16
###########################################################################################
#Load Data
#Code adapted from machinelearningmastery tutorials
###########################################################################################
# convert time series into supervised learning problem
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = math.sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores
# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
 
# load the dataset
dataframe = pd.read_csv(filename, usecols=[0,1], engine='python',infer_datetime_format=True, parse_dates=['DateTime'], index_col=['DateTime'])
hourly_groups = dataframe.resample('H')
hourly_data = hourly_groups.sum()
dataset = hourly_data.values
dataset = hourly_data.astype('float32')

#Split it 
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
traind, testd = dataset.iloc[0:train_size,:], dataset.iloc[train_size:len(dataset),:]

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
traind = scaler.fit_transform(traind)
testd = scaler.transform(testd)

#Turn the data into the supervised learning shape  

train,test = series_to_supervised(traind, n_lag, n_seq), series_to_supervised(testd, n_lag, n_seq)
train_values = train.values
test_values = test.values

# split into inputs and steps to predict
trainX,trainY= train_values[:, 0:n_lag], train_values[:, n_lag:]
testX,testY= test_values[:, 0:n_lag], test_values[:, n_lag:]
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1],1))
testX = numpy.reshape(testX, (testX.shape[0],testX.shape[1],1))

#model
model = Sequential()
model.add(LSTM(20, activation='tanh', input_shape=(n_lag, 1)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(n_seq))
model.compile(loss='mse', optimizer='adam')

# fit network
history=model.fit(trainX, trainY, validation_split=0.1, epochs=epochs, batch_size=batch_size, verbose=verbose)
#plot the evolutions of train 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('loss for test sets')
plt.xlabel('epoch')
plt.legend(['train','test'])
plt.show()

predicted = model.predict(testX)

#predicted = inverse_transform(predicted)
#testY = inverse_transform(testY)
score, scores = evaluate_forecasts(predicted,testY)
summarize_scores(name,score,scores)

#Plot the test results

test_hours_to_plot = 120
t0 = 20  # time to start plot of predictions
skip = 10  # skip prediction plots by specified hours
print('Plotting predictions...')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(testY[:test_hours_to_plot, 0] , label='Actual data')  # plot actual test series

# plot predicted values from t0 to t0+prediction_steps
plt.plot(np.arange(t0 - 1, t0 + n_seq), np.insert(predicted[t0, :], 0, testY[t0 - 1, 0]), color='red', label='t+{0} evolution of predictions'.format(n_seq))
for i in range(t0, test_hours_to_plot, skip):
    t0 += skip
    if t0 + n_seq> test_hours_to_plot:  # check plot does not exceed boundary
        break
    plt.plot(np.arange(t0 - 1, t0 + n_seq), np.insert(predicted[t0, :], 0, testY[t0 - 1, 0]), color='red')

    # plot predicted value of t+prediction_steps as series
plt.plot(predicted[:test_hours_to_plot , n_seq - 1] , label='t+{0} prediction series'.format(n_seq))
plt.legend(loc='best')
plt.ylabel('Power scaled')
plt.xlabel('Time in hours')
plt.title('Predictions for first {0} hours in test set'.format(test_hours_to_plot ))
plt.show()

