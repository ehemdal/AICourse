import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [ 'I love my dog',
              'I love my cat',  
              'I love my other cat',
              'You love my dog!',
              'I love my dog and my cat'
            ]
print("Sentences = ", sentences)
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index   
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=6, padding='post', truncating='post')
print("Word Index = ", word_index)
print("Sequences = ", sequences)
print("Padded Sequences = ", padded)
test_data = [ 'I really love my dog',
              'My dog loves my manatee and my cat'
            ]
test_seq = tokenizer.texts_to_sequences(test_data)
print("Test Sequences = ", test_seq)    
test_pad = pad_sequences(test_seq, maxlen=5)
print("Test Padded Sequences = ", test_pad)
