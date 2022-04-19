from flask import Flask, render_template,request
import pickle 
import numpy as np
model=pickle.load(open("Diabete_model.pkl",'rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('diabetes.html')

@app.route

@app.route('/prediction',methods=['post'])
def predict():
    height=float(request.form['height'])/100    
    weight=float(request.form['weight'])
    BMI=weight/(height*height)
    highBP=float(request.form['HighBP'])   
    walk_diff=float(request.form['Wdiff'])
    high_col=float(request.form['HighC'])
    gen_health=float(request.form['Ghealth'])
    arr=np.array([[highBP,BMI,walk_diff,gen_health,high_col]])
    pred=model.predict(arr)
    return render_template("result.html",data=pred[0])



if __name__=="__main__":
    app.run(debug=True)