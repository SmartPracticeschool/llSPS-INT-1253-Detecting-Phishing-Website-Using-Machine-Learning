import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
from sklearn.preprocessing import OneHotEncoder
oh=pickle.load(open('encoder.pkl','rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML GUI
    '''
    x_test=[[int(x) for x in request.form.values()]]
    x_test=oh.transform(x_test).toarray()
    prediction=model.predict(x_test)
    output=prediction[0]
    if(output==0):
        return render_template('index.html',prediction_text='It is A Good Website. No Fear about Phishing')
    else:
        return render_template('index.html',prediction_text='This site is suspicious . Probably a phishing site. Beware')

if __name__=="__main__":
    app.run(debug=True)
