# -*- coding: utf-8 -*-
"""
Created on Thu Jul 06 22:02:01 2017

@author: Jahangeer
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load dataset
dataset1 = numpy.loadtxt("Input.csv", delimiter=",")
dataset2 = numpy.loadtxt("Output.csv", delimiter=",")
# save into input (X) and output (Y) variables
X = dataset1
Y = dataset2
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dropout(.7))
model.add(Dense(7, activation='relu'))
model.add(Dropout(.7))
model.add(Dense(5, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='Nadam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=500, batch_size=100000)
# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#print("now testing against diffrent set of data")
#X = dataset1
#Y = dataset2
#scores = model.evaluate(X, Y)
#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))