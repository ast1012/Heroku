import pickle
from flask import Flask, request, render_template
#from werkzeug.wrappers import Request, Response
import numpy as np

app = Flask(__name__ , template_folder='template')
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #for redering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='The predicted score of student should be {}'.format(output))


if __name__=="__main__":
    app.run(debug=True)
    