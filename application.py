from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

#import ridge and standardscaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
sd_scaler=pickle.load(open('models/scaler.pkl','rb'))



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def data_point():
    if request.method=='POST':
        temp=float(request.form.get('temp'))
        rh=float(request.form.get('rh'))
        ws=float(request.form.get('ws'))
        rain=float(request.form.get('rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        new_data=sd_scaler.transform([[temp,rh,ws,rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data)

        return render_template('home.html',results=result[0])

        
    else:
        return render_template('home.html')

    

if __name__=="__main__":
    app.run(host="0.0.0.0")
