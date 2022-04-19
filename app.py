from asyncio.windows_events import NULL
from flask import Flask, render_template, request
import pickle
import numpy as np
#Models importation here 
model_diabetes=pickle.load(open("Diabete_model.pkl",'rb'))
model_covid=pickle.load(open("covid.pkl",'rb'))
model_heart=pickle.load(open("heart_model.pkl",'rb'))

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes_check')
def diabetes():
    return render_template('diabetes.html')

@app.route('/covid_check')
def corona():
    return render_template('corona.html')

@app.route('/results')
def result():
    return render_template('result.html')
@app.route('/heart_check')
def heart():
    return render_template('heart.html')
@app.route('/prediction_diabetes',methods=['post'])
def prediction_diabetes():
    height=float(request.form['height'])/100    
    weight=float(request.form['weight'])
    BMI=weight/(height*height)
    highBP=float(request.form['HighBP'])   
    walk_diff=float(request.form['Wdiff'])
    high_col=float(request.form['HighC'])
    gen_health=float(request.form['Ghealth'])
    #transform data if needed...
    arr=np.array([[highBP,BMI,walk_diff,gen_health,high_col]])
    pred=model_diabetes.predict(arr)
    #redirecting to the results page 
    return render_template("result.html",data=pred[0],sick=0)


@app.route('/prediction_covid',methods=['post'])
def prediction_covid():
    BrProblem=request.form['BrProblem']
    Fever=request.form['Fever']
    DryC=request.form['DryC']
    SoreT=request.form['SoreT']
    HyperT=request.form['HyperT']
    AbroadT=request.form['AbroadT']
    Contact=request.form['Contact']
    LargeG=request.form['LargeG']
    PublicE=request.form['PublicE']
    symptomes=[BrProblem,Fever,DryC,SoreT,HyperT,AbroadT,Contact,LargeG,PublicE]
    symptomes_binary=[]
    for i in symptomes:
        if i=='no' : symptomes_binary.append(0) 
        else : symptomes_binary.append(1)
    pred=model_covid.predict([symptomes_binary])
    #return "<h2>test</h2>"
    return render_template("result.html",data=pred[0],sick=1)

@app.route('/prediction_heart' ,methods=['post'])
def prediction_heart():
    smoke=int(request.form['smoke'])
    diabetic=int(request.form['diabetic'])
    stroke=int(request.form['stroke'])
    wdiff=int(request.form['Wdiff'])
    kidney=int(request.form['kidney'])
    physHlth=float(request.form['PhysHlth'])
    age=int(request.form['age'])
    physAct=int(request.form['physAct'])
    cancer=int(request.form['cancer'])
    symptomes=[[age,stroke,wdiff,physHlth,diabetic,kidney,smoke,physAct,cancer]]
    pred=model_heart.predict(symptomes)
    return render_template('result.html',data=pred[0],sick=2)
if __name__ == "__main__":
    app.run(debug=True)
