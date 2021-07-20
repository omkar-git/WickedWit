import numpy as np
import pandas as pd
from flask import Flask
from flask import request
from flask import render_template
import pickle

from model import original, numeric

y1 = pd.DataFrame({'ROLE':original, 'Associated Number':numeric})

print(y1)

app=Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route("/")
def home():
        return render_template("home.html")

@app.route("/index.html")
def index():
    return render_template("index.html") 

@app.route("/register.html")
def register():
    return render_template("register.html")   

@app.route("/blog.html")
def blog():
    return render_template("blog.html")        

@app.route("/career.html")
def career():
    return render_template("career.html")     


@app.route("/res")
def result():
    final_features = [['3','5','4','3','5','4','3','5','2','2','5','2','5','4','2','5','5','3','2','5','5','4','2','3','4']]
    p = model.predict(final_features)
    pred= "Prediction : {}".format(y1[y1['Associated Number']==p[0]]['ROLE'])
    # return pred[13]
    if(pred[13] == '0'):
        return render_template("0TechSup.html")
    if(pred[13] == '1'):
        return render_template("1SoftDev.html") 
    elif(pred[13] == '2'):
        return render_template("2UI.html") 
    elif(pred[13] == '3'):
            return render_template("3DataAna.html") 
    elif(pred[13] == '4'):
        return render_template("4TechWriter.html") 
    elif(pred[13] == '5'):
        return render_template("5WebDev.html") 
    elif(pred[13] == '6'):
        return render_template("6SoftTest.html") 
    elif(pred[13] == '7'):
        return render_template("7BusAnalyst.html") 
    else:
        return "Sorry <._.> Our server is currently busy... Please fill out form again or try again after sometime."
        


@app.route('/predict.html',methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    pred= "Prediction : {}".format(y1[y1['Associated Number']==prediction[0]]['ROLE'])
    if(pred[13] == '0'):
        return render_template("0TechSup.html")
    if(pred[13] == '1'):
        return render_template("1SoftDev.html") 
    elif(pred[13] == '2'):
        return render_template("2UI.html") 
    elif(pred[13] == '3'):
            return render_template("3DataAna.html") 
    elif(pred[13] == '4'):
        return render_template("4TechWriter.html") 
    elif(pred[13] == '5'):
        return render_template("5WebDev.html") 
    elif(pred[13] == '6'):
        return render_template("6SoftTest.html") 
    elif(pred[13] == '7'):
        return render_template("7BusAnalyst.html") 
    else:
        return "Sorry <._.> Our server is currently busy... Please fill out form again or try again after sometime."                


if __name__ == '__main__':
    app.run(debug=True)
