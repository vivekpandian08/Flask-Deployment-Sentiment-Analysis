import numpy as np
import pandas as pd
import pickle
from flask import Flask, jsonify, render_template, request

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import h5py
from keras.models import load_model

# load the dataset but only keep the top n words, zero the rest
top_words = 90000
max_words = 500

#load the csv file saved

df = pd.read_csv('C:/Users/vivek/OneDrive/Desktop/flaskapps/Sentiment_Analysis/food_delivery_reviews.csv', encoding='utf-8')

tokenizer_obj = Tokenizer(num_words=top_words)
tokenizer_obj.fit_on_texts(df['content'].values)

def pred(userreview):
    test_samples = [userreview]
    review_tokens = tokenizer_obj.texts_to_sequences(test_samples)
    review_tokens_pad = pad_sequences(review_tokens, maxlen=max_words)

    print("call predict")
    # Load in pretrained model
    loaded_model = load_model('C:/Users/vivek/OneDrive/Desktop/flaskapps/Sentiment_Analysis/Model/model.h5')
    print("Loaded model from disk")
    pred = loaded_model.predict(x=review_tokens_pad)
    predict_class = np.argmax(pred, axis=1)
    print(predict_class)
    if predict_class[0] == 0:
        sentiment_str = "Negative" 
    elif predict_class[0] == 1:
        sentiment_str = "Neutral" 
    elif predict_class[0]==2:
        sentiment_str = "Positive" 
    return sentiment_str

    #sentiment = loaded_model.predict(x=review_tokens_pad)
# webapp
app = Flask(__name__, template_folder='C:/Users/vivek/OneDrive/Desktop/flaskapps/Sentiment_Analysis/')


@app.route('/predict', methods=['POST'])
def prediction():
    
        message = request.form['message']
        print(message)
        response =  pred(message)
        print(response)
        return jsonify(response)


@app.route('/C:/Users/vivek/OneDrive/Desktop/flaskapps/Sentiment_Analysis/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)