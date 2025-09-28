import nltk
import csv
import string
from bs4 import BeautifulSoup
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
