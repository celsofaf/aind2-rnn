import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
    y = series[window_size:]
    
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X, y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    
    return model


### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    wanted = set(punctuation).union(set(letters))
    unwanted = set(text).difference(wanted)
    
    for char in unwanted:
        text = text.replace(char, ' ')
        text = text.replace('  ', ' ')  #double spaces

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    j = 0
    while j + window_size + 1 <= len(text):
    #while j + window_size < len(text):
        inputs.append(text[j:j+window_size])
        outputs.append(text[j+window_size])
        j = j + step_size
    
    # reshape each 
#    inputs = np.asarray(inputs)
#    inputs.shape = (np.shape(inputs)[0:2])
#    outputs = np.asarray(outputs)
#    outputs.shape = (len(outputs),1)

    return inputs, outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    #model.add(Dense(num_chars))
    model.add(Dense(num_chars, activation='softmax'))
    
    return model
