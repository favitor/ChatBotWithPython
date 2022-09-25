import nltk
#nltk.download()
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('intents.json').read()
intents = json.loads(data_file)


for intent in intents['intents']:
	for pattern in intent['patterns']:
		#Take each word and tokezine it
		w = nltk.word_tokenize(pattern)
		words.extend(w)
		#adding documents
		documents.append((w, intent['tag']))

		#adding classes to class list
		if intent['tag'] not in classes:
			classes.append(intent['tag'])


words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), 'documents')
print(len(classes), 'classes', classes)
print(len(words), 'Unique lemmatized words', words)

pickle.dump(words, open('words.plk', 'wb'))
pickle.dump(classes, open('classes.plk', 'wb'))


#Initializing training data
training = []
output_empty = [0] * len(classes)
for doc in documents:
	#Initializing bag of words
	bag = []
	pattern_words = doc[0]
	pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

	#Create bag of words array with 1, if word match found in current pattern
	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)


	#output
	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1

	training.append([bag, output_row])


#Shuffle features and turn into arrays
random.shuffle(training)
training = np.array(training)

#create train test lists
train_x = list(training[:,0])
train_y = list(training[:,1])


#create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Compile model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#fit and save model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',)