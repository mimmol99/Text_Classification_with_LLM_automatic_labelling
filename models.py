import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense, GlobalMaxPool1D, Bidirectional
from transformers import BertTokenizer, TFBertForSequenceClassification, AdamW
import tensorflow as tf
import joblib
import json
from tensorflow.keras.layers import *#Conv2D,Flatten,Dense,Dropout,RandomFlip,RandomZoom,RandomRotation,Layer,Resizing
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow_hub as hub

# explicitly require this experimental feature
from sklearn.experimental import enable_halving_search_cv # noqa
# now you can import normally from model_selection
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import HalvingRandomSearchCV

class BaseModel:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

class KNNModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = KNeighborsClassifier(n_neighbors=5)

    def train(self):
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.model.fit(X_train_tfidf, self.y_train)
        joblib.dump(self.model, 'knn_model.pkl')
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

    def halving_random_train(self):
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        param_grid = {"n_neighbors" : [1,3,5,7],"weights":["uniform","distance",None],"algorithm":['auto','ball','kd_tree','brute']}
        search = HalvingRandomSearchCV(self.model, param_grid,refit=True,verbose=1)
        self.model = search.best_estimator_
        #self.model.fit(X_train_tfidf,**search.get_params())
        joblib.dump(self.model, 'knn_model.pkl')
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')
      
    def eval(self):
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        preds = self.model.predict(X_test_tfidf)
        return accuracy_score(self.y_test, preds)

    def predict(self, texts):
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)

class RFModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test):
        super().__init__(X_train, y_train, X_test, y_test)
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self):
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        self.model.fit(X_train_tfidf, self.y_train)
        joblib.dump(self.model, 'rf_model.pkl')
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

    def halving_random_train(self):
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        param_grid = {"n_estimators" : [10,100,1000],"weights":["uniform","distance",None],"criterion":['gini','entropy','log_loss'],"max_depth":[None,10,100]}
        search = HalvingRandomSearchCV(self.model, param_grid, resource='n_samples',refit=True,verbose=1).fit(self.X_train,self.y_train)
        self.model = search.best_estimator_
        #self.model.fit(X_train_tfidf,**search.get_params())
        joblib.dump(self.model, 'rf_model.pkl')
        joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')

    def eval(self):
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        preds = self.model.predict(X_test_tfidf)
        return accuracy_score(self.y_test, preds)

    def predict(self, texts):
        X_tfidf = self.vectorizer.transform(texts)
        return self.model.predict(X_tfidf)

class LSTMModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test, max_words=5000, max_len=100, embedding_dim=128):
        super().__init__(X_train, y_train, X_test, y_test)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
        self.max_len = max_len
        self.model = Sequential()
        self.model.add(Embedding(max_words, embedding_dim, input_length=max_len))
        self.model.add(Bidirectional(LSTM(64, return_sequences=True)))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self):
        self.tokenizer.fit_on_texts(self.X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=self.max_len)
        self.model.fit(X_train_pad, self.y_train, epochs=5, batch_size=32, validation_split=0.2)
        save_model(self.model, 'lstm_model.h5')
        tokenizer_config = self.tokenizer.to_json()
        with open('lstm_tokenizer.json', 'w') as f:
            json.dump(tokenizer_config, f)

    def eval(self):
        X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=self.max_len)
        loss, accuracy = self.model.evaluate(X_test_pad, self.y_test)
        return accuracy

    def predict(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.max_len)
        return (self.model.predict(pad) > 0.5).astype("int32")

class GRUModel(BaseModel):
    def __init__(self, X_train, y_train, X_test, y_test, max_words=5000, max_len=100, embedding_dim=128):
        super().__init__(X_train, y_train, X_test, y_test)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
        self.max_len = max_len
        self.model = Sequential()
        self.model.add(Embedding(max_words, embedding_dim, input_length=max_len))
        self.model.add(Bidirectional(GRU(64, return_sequences=True)))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self):
        self.tokenizer.fit_on_texts(self.X_train)
        X_train_seq = self.tokenizer.texts_to_sequences(self.X_train)
        X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=self.max_len)
        self.model.fit(X_train_pad, self.y_train, epochs=5, batch_size=32, validation_split=0.2)
        save_model(self.model, 'gru_model.h5')
        tokenizer_config = self.tokenizer.to_json()
        with open('gru_tokenizer.json', 'w') as f:
            json.dump(tokenizer_config, f)
            
    def eval(self):
        X_test_seq = self.tokenizer.texts_to_sequences(self.X_test)
        X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=self.max_len)
        loss, accuracy = self.model.evaluate(X_test_pad, self.y_test)
        return accuracy

    def predict(self, texts):
        seq = self.tokenizer.texts_to_sequences(texts)
        pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=self.max_len)
        return (self.model.predict(pad) > 0.5).astype("int32")


class HubLayer(Layer):
    def __init__(self):
        super().__init__()
        self.preprocessor = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            trainable=False
        )
        self.encoder = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
            trainable=True
        )

    def call(self, inputs):
        preprocessed_text = self.preprocessor(inputs)
        outputs = self.encoder(preprocessed_text)
        return outputs['pooled_output']

class BERT(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.hub_layer = HubLayer()
        self.dense_32 = tf.keras.layers.Dense(32, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense_16 = tf.keras.layers.Dense(16, activation='relu')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.hub_layer(inputs)
        x = self.dense_32(x)
        x = self.dropout(x)
        #x = self.dense_16(x)
        #x = self.dropout(x)
        return self.output_layer(x)
    
class BERTModel(BaseModel):
    def __init__(self, train_dataset):
        self.model = BERT()
        total_batches = tf.data.experimental.cardinality(train_dataset).numpy()
        train_size = int(0.8 * total_batches)
        val_size = int(0.2 * total_batches)

        self.train_dataset = train_dataset.take(train_size)
        self.val_dataset = train_dataset.skip(train_size).take(val_size)

    def train(self):
        epochs = 50
        patience = epochs // 10

        early_stopping = EarlyStopping(
            monitor='loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=patience
        )

        self.model.compile(
            optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        self.model.summary()

        history = self.model.fit(
            self.train_dataset,
            validation_data=self.val_dataset,
            epochs=epochs,
            verbose=1,
            callbacks=[early_stopping]
        )

    def eval(self, test_dataset):
        results = self.model.evaluate(test_dataset)
        print(f"Test Loss: {results[0]}")
        print(f"Test Accuracy: {results[1]}")
        print(f"Test Precision: {results[2]}")
        print(f"Test Recall: {results[3]}")
        return results

    def predict(self, texts):
        text_dataset = tf.data.Dataset.from_tensor_slices(texts).batch(1)
        predictions = self.model.predict(text_dataset)
        return (predictions > 0.5).astype("int32")

