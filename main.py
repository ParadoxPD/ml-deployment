import pickle
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    area = request.form.get('area')
    x = []
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(x)
    prediction = model.predict([scaled_X])
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text=f'Price = {output}')


if __name__ == '__main__':
    app.run(debug=True)
