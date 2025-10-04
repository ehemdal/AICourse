import nltk
import csv
import string
import numpy as np
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Function to plot training and validation graphs
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

# Flag for enabling/disabling debugging output
DEBUG = True 

# Download the stopwords from NLTK
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords = stopwords.words('english')

# Create a translation table to remove punctuation. maketrans creates a mapping table
# The first two arguments are empty strings because we are not replacing any characters
# The third argument is a string of characters to delete
# This table is used in the loop below to remove punctuation
table = str.maketrans('', '', string.punctuation)

sentences=[]
labels=[]

# Read the CSV file and process each row
# The replace() calls add spaces around punctuation to ensure they are treated as separate "words"
with open('./binary-emotion.csv', encoding='UTF-8') as csvfile:
  reader = csv.reader(csvfile, delimiter=",")
  for row in reader:
    labels.append(int(row[0]))
    sentence = row[1].lower()
    sentence = sentence.replace(",", " , ")
    sentence = sentence.replace(".", " . ")
    sentence = sentence.replace("-", " - ")
    sentence = sentence.replace("/", " / ")
    sentence = sentence.replace("'", " ' ")
    sentence = sentence.replace('"', ' " ')

    # Use BeautifulSoup to remove HTML tags
    # Since I have Python 3, I need to install package beautifulsoup4 to get it 
    soup = BeautifulSoup(sentence, features="html.parser")
    sentence = soup.get_text()

    # Split the sentence into words, remove punctuation, and filter out stopwords
    words = sentence.split()
    filtered_sentence = ""
    for word in words:
        word = word.translate(table)
        if word not in stopwords:
            filtered_sentence = filtered_sentence + word + " "
    # Uncomment this line to see all 35327 sentences.
    # print(filtered_sentence)
    sentences.append(filtered_sentence)
    
print("There are " + str(len(labels)) + " labels in the dataset")
print("There are " + str(len(sentences)) + " sentences in the dataset")

# Partition the data into training and testing sets
# Use the first 28000 for training and the rest for testing
training_size = 28000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
print("There are " + str(len(training_sentences)) + " training sentences")
print("There are " + str(len(testing_sentences)) + " testing sentences")

vocab_size = 20000
embedding_dim = 32
max_length = 10
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index
print("There are " + str(len(word_index)) + " unique words in the dataset")
training_sequences = tokenizer.texts_to_sequences(training_sentences)
print("There are " + str(len(training_sequences)) + " training sequences")
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print("Training Padded Sequences= ", training_padded)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print("There are " + str(len(testing_sequences)) + " testing sequences")
print("Testing Padded Sequences= ", testing_padded)

if DEBUG:
  # Print the word counts for debugging purposes...this generates a lot of output
  wc=tokenizer.word_counts
  print(wc)
  print("There are " + str(len(wc)) + " words in the word count dictionary")


# Because I installed TensorFlow 2, I need to turn my lists into numpy arrays
# I do it here because it's easier to remove punctuation and stopwords from simple Python lists
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# Here's the model....
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(8, kernel_regularizer=tf.keras.regularizers.l2(0.025), activation='relu'))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

num_epochs = 100
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)

# Plot training and validation graphs
if DEBUG:
  plot_graphs(history, "accuracy")  
  plot_graphs(history, "loss")
  model.evaluate(testing_padded, testing_labels)

# Try out some sentences to see how well the model detects sarcasm
# The model should return a number close to 0 for non-sarcastic sentences
# and a number close to 1 for sarcastic sentences....the probability that the sentence is sarcastic.
sentences = ["I'm really upset right now and not happy with you! ANGRY!", 
             "She said yes! We're getting married! Wow!", 
             "I love trash!", 
             "I am so sad and depressed.",
             "Another flat tire, this day keeps getting better and better.", 
             "I am furious about what you did!"]
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(model.predict(padded))