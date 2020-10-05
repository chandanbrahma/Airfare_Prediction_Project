# -*- coding: utf-8 -*-
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
import pandas as pd
import datetime
import pickle
from flask import Markup
import numpy as np
from pmdarima import auto_arima

# Load the Arima model for domestic
filename = 'dom_air-fare-prediction-model.pkl'
dom_predict = pickle.load(open(filename, 'rb'))

# Load the Arima model for international
filename2 = 'intl_air-fare-prediction-model.pkl'
intl_predict = pickle.load(open(filename2, 'rb'))

app = Flask(__name__,template_folder='templates')
 
@app.route("/")
def home():    
    return render_template('DAQ.html')

@app.route("/", methods=['GET', 'POST'])
def predict():    
    if request.method == 'POST':
        # Get the input from post request
        datevalue=request.form['ddate']
        date = datetime.datetime.strptime(datevalue, '%Y-%m-%d').date()
        day = date.day
        #data = day 
        #prediction=0
        #predcition2 = 0
        option = request.form['options']
        if option == 'Domestic':
            #prediction=0
            prediction=dom_predict.predict(day)
            output=list(prediction)  
            #print(output)
            return render_template('DAQ.html',prediction=output[0])
        else:
            #predcition2 = 0
            prediction2=intl_predict.predict(day)
            output2=list(prediction2)  
            #print(output2)
            return render_template('DAQ.html',prediction=output2[0])
        
        """
        df=pd.DataFrame(output,columns = ['Prediction'])      
        df["Date"]=timestamps
        table=df.to_html(escape=False)
        table=Markup(table)
        print("End of def")      
        return render_template('DAQ.html',prediction=table) """    
      

if __name__ == '__main__':  
   app.run(debug = True)  