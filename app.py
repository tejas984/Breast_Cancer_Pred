from flask import Flask,render_template,request,redirect
import pickle
import numpy as np

model=pickle.load(open("model.pkl","rb"))

app=Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict_Cancer():

    float_features = [float(x) for x in request.form.values()]
    result=model.predict([np.array(float_features)])
    
    if result==1:
        return "<h1 style='color:green'>Malignant</h1>"
    else:
        return "<h1 style='color:red'>benign</h1>"
    
    
if __name__=="__main__":
    app.run(host="127.0.0.1",debug=True,port=5001)