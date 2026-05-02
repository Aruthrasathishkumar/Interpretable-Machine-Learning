import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential # If you get an error it's because in newer versions of keras,
# Sequential is under keras not keras.models. So you should import it like: from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences # You can also use: from tensorflow.keras.utils import pad_sequences


max_number_words = 50000 # The maximum number of words in your lexicon or vocabulary.
max_phrase_length = 70 # Max number of words in each phrase or document.
embedding_dim = 100  # This is the size of the vector space in which words will be embedded.
LSTM_units = 128
epochs = 15
batch_size = 64

df = pd.read_csv(r'D:\aruth\Documents\AIT 636 ML\Project_14\Movie Reviews.csv')
print(df.head())# Let's create a bar chart to see the number of samples in each class
# sns.countplot(data=df, x='label')
# plt.xlabel('Label')
# plt.title('Size of classes')
# plt.show()

# # If the above code does show both columns or has a problem, use the code below to create the bar chart
# df['label'].value_counts().plot(kind='bar')
# plt.show()

LSTM_units_list = [64, 128, 256]
embedding_dim_list = [75, 100]
max_phrase_length_list = [50, 75, 100]

for LSTM_units in LSTM_units_list:
    for embedding_dim in embedding_dim_list:
        for max_phrase_length in max_phrase_length_list:
            # We use a tokenizer to convert textual phrases to vectors of integers.
            tokenizer = Tokenizer(num_words=max_number_words, # The maximum number of words in your lexicon or vocabulary.
                                  filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', # It will remove these characters from your messages.
                                  lower=True) # It will lower case everything.
            tokenizer.fit_on_texts(df['text'].values) # It will create a lexicon based on the entire dataset.

            # Print the size of lexicon
            print('Found %s unique tokens.' % len(tokenizer.word_index))

            x = tokenizer.texts_to_sequences(df['text'].values)
            x = pad_sequences(x, maxlen=max_phrase_length)
            print('Shape of data tensor:', x.shape)

            # Converting categorical labels to vectors.
            y = pd.get_dummies(df['label']).values
            print('Shape of label tensor:', y.shape)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
            print(x_train.shape, y_train.shape)
            print(x_test.shape, y_test.shape)

            model = Sequential() # The Sequential model is a linear stack of layers. You can create a Sequential model by adding layers:
            model.add(Embedding( # The role of embedding is to learn a representation (vector) for each word.
                                input_shape=(max_phrase_length,), # This is the length of the input vectors.
                                input_dim=max_number_words, # This is the size of the vocabulary in the text data.
                                output_dim=embedding_dim))
            model.add(SpatialDropout1D(0.1))
            model.add(LSTM(units=LSTM_units, activation='relu',
                           dropout=0.2, recurrent_dropout=0.2))

            model.add(Dense(units=128, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(units=2, activation='softmax'))
            print(model.summary())

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

            # Evaluate the model
            test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

            print("Test Loss:", test_loss)
            print("Test Accuracy:", test_accuracy)

            # Create a graph of the training and validation accuracy over the training epochs
            plt.figure(figsize=(8, 5))
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            plt.plot(history.history['val_accuracy'], label='Test Accuracy')
            plt.xlabel('Epoch Number')
            plt.ylabel('Accuracy')
            plt.title(
                'Accuracy vs Epochs\n'
                + 'LSTM_units=' + str(LSTM_units)
                + ', embedding_dim=' + str(embedding_dim)
                + ', max_phrase_length=' + str(max_phrase_length)
            )
            plt.legend()
            plt.show()

