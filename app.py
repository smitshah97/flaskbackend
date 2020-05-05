from flask import Flask,Blueprint, jsonify, request, redirect
import sys
import pandas as pd
import pymongo
import json
import io
import simplejson
from flask_request_params import bind_request_params
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFpr, chi2
from sklearn.feature_selection import f_regression, mutual_info_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
from kmodes.kmodes import KModes
#from . import db 
#from .models import Movie
from flask_cors import CORS, cross_origin
from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper



application=app = Flask(__name__)
app.before_request(bind_request_params)

cors = CORS(app, resources={r"/execute_clusteringopearation": {"origins": "http://13.232.254.221:80"}})
app.config['CORS_HEADERS'] = 'Content-Type'


#app.config["FILE_UPLOADS"]='C:\Users\SMIT SHAH\Desktop\ampleai\flaskbackend\api\uploads'


@app.route("/")
def index():
    
    return 'AWS FLASK DEPLOYED', 201


@app.route('/sampleapi', methods=['GET'])
def add_movie():
   
    return 'Done', 201


@app.route("/add_csv", methods=['POST']) 
@cross_origin()
def uploaddataset():
    if request.method=="POST":
        if request.files:
            print("ABCDEF")
            file=request.files["files[]"]
        a="Successfully Submitted in Flask"
        mng_client = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db = mng_client['ampleai'] 
        collection_name = 'csv' 
        db_cm = mng_db[collection_name]
        print(file)
        s=file.read()
        csvtable=pd.read_csv(io.StringIO(s.decode('utf-8')))
        #print(csvtable)
        db_cm.remove()
        #
        numericcolumnlist=[]
        int64list=list(csvtable.select_dtypes(include=["int64"]).columns)
        numericcolumnlist.extend(int64list)
        int32list=list(csvtable.select_dtypes(include=["int32"]).columns)
        numericcolumnlist.extend(int32list)
        float64list=list(csvtable.select_dtypes(include=["float64"]).columns)
        numericcolumnlist.extend(float64list)
        float32list=list(csvtable.select_dtypes(include=["float32"]).columns)
        numericcolumnlist.extend(float32list)
        #
        data_dict = csvtable.to_dict("records")
        db_cm.insert({"User":"Smit","data":data_dict,"numericcolumnlist":numericcolumnlist,"s":s},check_keys=False)
        #data_from_db = db_cm.find_one({"abc":"Smit"})
        #df = pd.DataFrame(data_from_db["data"])
        #print(df)
        #df1=data_from_db["data"]
        #print(df1[0])
        #return jsonify({'df1' : df1})
    return a

@app.route("/fetch_csv", methods=['GET']) 
@cross_origin()
def fetchdataset():
    if request.method=="GET":
        mng_client1 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db1 = mng_client1['ampleai'] 
        collection_name1 = 'csv' 
        db_cm1 = mng_db1[collection_name1]
        data_from_db = db_cm1.find_one({"User":"Smit"})
        s=data_from_db["s"]
        df1=data_from_db["data"]
        
       
        if type(s)==str:
            csvtable=pd.DataFrame.from_dict(df1)
        else:
            csvtable=pd.read_csv(io.StringIO(s.decode('utf-8')))
        
        print("sdfgfsdfgdgf")
        

        for dict in df1:
            keysreference=dict.keys()
        keysreference=list(keysreference)
        keylist=[]
        i=0
        lenkeys=len(keysreference)
        for element in keysreference:
            keylist.append({"label":element,"field": element,"sort": 'asc',"width": 200})
            #if i<1:
                #keylist[i]["fixed"]="left"
            #elif i==lenkeys-1:
                #keylist[i]["fixed"]="right"
            i=i+1    
        i=0
        valuelist=[]
        for dict in df1:
            valuelist.append({"key":i})
            j=0
            for keyreference in keysreference:
                valuelist[i][keyreference]=list(dict.values())[j]
                j=j+1
            i=i+1    
        datasource={"columns":keylist,"rows":valuelist}
        return simplejson.dumps({"datasource":datasource,"keylist":keylist,"df1":df1,"valuelist":valuelist},ignore_nan=True)
        
    return a


@app.route("/fetch_columnarithmeticoperation", methods=['GET']) 
@cross_origin()
def fetchcolumns():
    if request.method=="GET":
        mng_client1 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db1 = mng_client1['ampleai'] 
        collection_name1 = 'csv' 
        db_cm1 = mng_db1[collection_name1]
        data_from_db = db_cm1.find_one({"User":"Smit"})
        dictfromdatabase=data_from_db["data"]
        csvtable=pd.DataFrame.from_dict(dictfromdatabase)
        #
        numericcolumnlist=[]
        int64list=list(csvtable.select_dtypes(include=["int64"]).columns)
        numericcolumnlist.extend(int64list)
        int32list=list(csvtable.select_dtypes(include=["int32"]).columns)
        numericcolumnlist.extend(int32list)
        float64list=list(csvtable.select_dtypes(include=["float64"]).columns)
        numericcolumnlist.extend(float64list)
        float32list=list(csvtable.select_dtypes(include=["float32"]).columns)
        numericcolumnlist.extend(float32list)

        return jsonify({"keysreference":numericcolumnlist})
        
    return a


@app.route("/execute_columnarithmeticoperation", methods=['POST']) 
@cross_origin()
def executecolumns():
    if request.method=="POST":
        if request.files:
            print("ABCDEF")
        data = request.get_json(silent=True)
        item = {'values': data.get('values')}
        print(item)
        a="Successfully Submitted in Flask"
        mng_client2 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db2 = mng_client2['ampleai'] 
        collection_name2 = 'csv' 
        db_cm2 = mng_db2[collection_name2]
        data_from_db = db_cm2.find_one({"User":"Smit"})
        
        dictfromdatabase=data_from_db["data"]
        df2=pd.DataFrame.from_dict(dictfromdatabase)
        print(df2.head())
        # Aithmetic Oeration on Column
        if item["values"]["select"]=="Subtract":
            for column in item["values"]["select-multiple"]:
                df2[column]=df2[column].apply(lambda x: x - item["values"]["input-number"])
        if item["values"]["select"]=="Add":
            for column in item["values"]["select-multiple"]:
                df2[column]=df2[column].apply(lambda x: x + item["values"]["input-number"])
        if item["values"]["select"]=="Multiply":
            for column in item["values"]["select-multiple"]:
                df2[column]=df2[column].apply(lambda x: x * item["values"]["input-number"])
        if item["values"]["select"]=="Divide":
            for column in item["values"]["select-multiple"]:
                df2[column]=df2[column].apply(lambda x: x / item["values"]["input-number"])
        # Store Updated dictionary in database
        db_cm2.remove()
        #
       
        #
        data_dict = df2.to_dict("records")
        s1=df2.to_csv(index=False)
        db_cm2.insert({"User":"Smit","data":data_dict,"s":s1},check_keys=False)
    return a




@app.route("/fetch_allcolumns", methods=['GET']) 
@cross_origin()
def fetchallcolumns():
    if request.method=="GET":
        mng_client1 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db1 = mng_client1['ampleai'] 
        collection_name1 = 'csv' 
        db_cm1 = mng_db1[collection_name1]
        data_from_db = db_cm1.find_one({"User":"Smit"})
        dictfromdatabase=data_from_db["data"]
        dataframe=pd.DataFrame.from_dict(dictfromdatabase)
        fullcolumnslist=list(dataframe.columns)
        return jsonify({"fullcolumnslist":fullcolumnslist})
        
    return a


@app.route("/execute_splitcharoperation", methods=['POST']) 
@cross_origin()
def executesplitchar():
    if request.method=="POST":
        if request.files:
            print("ABCDEF")
        data = request.get_json(silent=True)
        item = {'values': data.get('values')}
        print(item)
        a="Successfully Submitted in Flask"
        mng_client2 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db2 = mng_client2['ampleai'] 
        collection_name2 = 'csv' 
        db_cm2 = mng_db2[collection_name2]
        data_from_db = db_cm2.find_one({"User":"Smit"})
        dictfromdatabase=data_from_db["data"]
        df2=pd.DataFrame.from_dict(dictfromdatabase)
        print(df2.head())
        
        
        #Split Operation on columns
        fetchedcolumns=item["values"]["select-multiple"]
        try:
            value=item["values"]["inputvalue"]
        except KeyError:
            value=" "
        if value=="":
            value=" "      
        print(value)
        up=value.upper()
        lo=value.lower()
        for fetchedcolumn in fetchedcolumns:
            occur=0
            for row in df2[fetchedcolumn]:
                count=0
                for letter in row:
                    if letter==value or letter==up or letter==lo:
                        count=count+1
                if count>occur:
                    occur=count
            numofsplits=occur+1
            splittedcolumns=[]
            for split in range(1,numofsplits+1):
                strsplit=str(split)
                columnname=fetchedcolumn + strsplit
                splittedcolumns.append(columnname)
            df2[splittedcolumns] = df2[fetchedcolumn].str.split(value,expand=True) 
        # Store Updated dictionary in database
        s=df2.to_csv()
        df2.to_csv("Titanic.csv",index=False)
        data_dict = df2.to_dict("records")
        db_cm2.remove()
        db_cm2.insert({"User":"Smit","data":data_dict,"s":s},check_keys=False)
    return a


@app.route("/fetch_stringcolumns", methods=['GET']) 
@cross_origin()
def fetchstringcolumns():
    if request.method=="GET":
        mng_client1 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db1 = mng_client1['ampleai'] 
        collection_name1 = 'csv' 
        db_cm1 = mng_db1[collection_name1]
        data_from_db = db_cm1.find_one({"User":"Smit"})
        dictfromdatabase=data_from_db["data"]
        csvtable=pd.DataFrame.from_dict(dictfromdatabase)
        #
        stringcolumnlist=[]
        objectlist=list(csvtable.select_dtypes(include=["object"]).columns)
        stringcolumnlist.extend(objectlist)
        

        return jsonify({"keysreference":stringcolumnlist})
        
    return a


@app.route("/execute_meltcolumnsoperation", methods=['POST']) 
@cross_origin()
def executemeltcolumns():
    if request.method=="POST":
        if request.files:
            print("ABCDEF")
        data = request.get_json(silent=True)
        item = {'values': data.get('values')}
        print(item)
        a="Successfully Submitted in Flask"
        mng_client2 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db2 = mng_client2['ampleai'] 
        collection_name2 = 'csv' 
        db_cm2 = mng_db2[collection_name2]
        data_from_db = db_cm2.find_one({"User":"Smit"})
        dictfromdatabase=data_from_db["data"]
        df2=pd.DataFrame.from_dict(dictfromdatabase)
        print(df2.head())
        
        
        #Melt Operation on columns
        fetchedcolumns=item["values"]["select-multiple"]
        df2=pd.melt(df2, id_vars=fetchedcolumns, var_name='Melted Column Values', value_name='Melted Row Values')
             
            
        # Store Updated dictionary in database
        s=df2.to_csv()
        df2.to_csv("Titanic.csv",index=False)
        data_dict = df2.to_dict("records")
        db_cm2.remove()
        db_cm2.insert({"User":"Smit","data":data_dict,"s":s},check_keys=False)
    return a


@app.route("/execute_regressionoperation", methods=['POST']) 
@cross_origin()
def executeregression():
    if request.method=="POST":
        if request.files:
            print("ABCDEF")
        data = request.get_json(silent=True)
        item = {'values': data.get('values')}
        print(item)
        a="Successfully Submitted in Flask"
        mng_client2 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db2 = mng_client2['ampleai'] 
        collection_name2 = 'csv' 
        db_cm2 = mng_db2[collection_name2]
        data_from_db = db_cm2.find_one({"User":"Smit"})
        dictfromdatabase=data_from_db["data"]
        df2=pd.DataFrame.from_dict(dictfromdatabase)
        stringcolumnlist=[]
        objectlist=list(df2.select_dtypes(include=["object"]).columns)
        stringcolumnlist.extend(objectlist)
        df2.drop(stringcolumnlist,axis=1,inplace=True)
        print(df2.head())
        # Our operation
        selectedtarget=item["values"]["select"]
        print(selectedtarget)
        df2[[selectedtarget]]
        X = df2.drop(selectedtarget,axis=1).values
        y = df2[selectedtarget].values
        Xfeature = df2.drop(selectedtarget,axis=1)
        yfeature = df2[selectedtarget]
        reg1 = GradientBoostingRegressor(random_state=1, n_estimators=10)
        reg2 = RandomForestRegressor(random_state=1, n_estimators=10)
        reg3 = LinearRegression()
        regensemble = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
        print(X.shape)
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        sel.fit_transform(X)
        print(X.shape)
        r2scorelist=[]
        i=2
        lenofX=len(df2.drop(selectedtarget,axis=1).columns)
        while i<=lenofX:
            X_new = SelectKBest(score_func=f_regression, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            ereg = regensemble.fit(X_train, y_train)
            y_pred = ereg.predict(X_test)
            r2scorelist.append(r2_score(y_test, y_pred))
            i=i+1
        clf1r2scorelist=[]
        i=2
        while i<=lenofX:
            X_new = SelectKBest(score_func=f_regression, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            trainedreg1 = reg1.fit(X_train, y_train)
            y_pred = trainedreg1.predict(X_test)
            clf1r2scorelist.append(r2_score(y_test, y_pred))
            i=i+1
        clf2r2scorelist=[]
        i=2
        while i<=lenofX:
            X_new = SelectKBest(score_func=f_regression, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            trainedreg2 = reg2.fit(X_train, y_train)
            y_pred = trainedreg2.predict(X_test)
            clf2r2scorelist.append(r2_score(y_test, y_pred))
            i=i+1
        clf3r2scorelist=[]
        i=2
        while i<=lenofX:
            X_new = SelectKBest(score_func=f_regression, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            trainedreg3 = reg3.fit(X_train, y_train)
            y_pred = trainedreg3.predict(X_test)
            clf3r2scorelist.append(r2_score(y_test, y_pred))
            i=i+1

        # Create data for comparison plot
        dataforcomparisonplot=[]
        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=clf1r2scorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"GradientBoostingRegressor"})
        
        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=clf2r2scorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"RandomForestRegressor"})
        
        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=clf3r2scorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"LinearRegression"})
        

        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=r2scorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"Ensemble"})
        
        print(dataforcomparisonplot)
        print(clf1r2scorelist)
        largestr2=max(r2scorelist)
        print(r2scorelist.index(largestr2))
        Optimalk=r2scorelist.index(largestr2) + 2
        print(Optimalk)
        X_new = SelectKBest(score_func=f_regression, k=Optimalk).fit_transform(X, y)
        print(X_new[0])
        print(X[0])
        firstrow=X[0].tolist()
        print(type(X[0]))
        Xcolumns=[]
        for col in Xfeature.columns:
            Xcolumns.append(col)
        print(Xcolumns)
        featurelist=[]
        for element in X_new[0]:
            print(firstrow.index(element))
            featurelist.append(Xcolumns[firstrow.index(element)])
        print(featurelist)
        # Predicting model 
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
        ereg = regensemble.fit(X_train, y_train)
        y_pred = ereg.predict(X_test)
        
        
        # Make new Dictionary to sore in database
        regressionmodel={}
        regressionmodel["dataforcomparisonplot"]=dataforcomparisonplot
        regressionmodel["Ensemble"]="Yes"
        regressionmodel["Type"]="Voting Regressor"
        regressionmodel["Estimators"]=["GradientBoostingRegressor","RandomForestRegressor","LinearRegression"]
        regressionmodel["numfeatures"]=Optimalk
        regressionmodel["featurenames"]=featurelist
        regressionmodel["featureselection"]=["Variance Threshlod","Univariate Selection"]
        # Metrics
        regressionmodel["Explained Variance Score"]=explained_variance_score(y_test, y_pred)
        regressionmodel["MeanAbsoluteError"]=mean_absolute_error(y_test, y_pred)
        regressionmodel["MeanSquaredError"]=mean_squared_error(y_test, y_pred)
        regressionmodel["RootMeanSquaredError"]=mean_squared_error(y_test, y_pred, squared=False)
        regressionmodel["MeanSquaredLogError"]=mean_squared_log_error(y_test, y_pred)
        regressionmodel["MedianAbsoluteError"]=median_absolute_error(y_test, y_pred)
        # Store Updated dictionary in database
        db_cm2.remove()
        #
       
        #
        data_dict = df2.to_dict("records")
        s1=df2.to_csv(index=False)
        db_cm2.insert({"User":"Smit","data":data_dict,"s":s1,"regressionmodel":regressionmodel},check_keys=False)
    return a

@app.route("/fetch_regressionmodel", methods=['GET']) 
@cross_origin()
def fetchregressionmodel():
    if request.method=="GET":
        mng_client1 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db1 = mng_client1['ampleai'] 
        collection_name1 = 'csv' 
        db_cm1 = mng_db1[collection_name1]
        data_from_db = db_cm1.find_one({"User":"Smit"})
        s=data_from_db["s"]
        df1=data_from_db["data"]
        regressionmodel=data_from_db["regressionmodel"]
        dataforcomparisonplot=regressionmodel["dataforcomparisonplot"]
        Ensemble=regressionmodel["Ensemble"]
        Type=regressionmodel["Type"]
        Estimators=regressionmodel["Estimators"]
        numfeatures=regressionmodel["numfeatures"]
        featurenames=regressionmodel["featurenames"]
        featureselection=regressionmodel["featureselection"]
        Variance_Threshold=featureselection[0]
        Univariate_selection=featureselection[1]
        MeanAbsoluteError=regressionmodel["MeanAbsoluteError"]
        MeanSquaredError=regressionmodel["MeanSquaredError"]
        RootMeanSquaredError=regressionmodel["RootMeanSquaredError"]
        MeanSquaredLogError=regressionmodel["MeanSquaredLogError"]
        MedianAbsoluteError=regressionmodel["MedianAbsoluteError"]
        dataforerrormetric=[]
        dataforerrormetric.append({"Metric":"MeanAbsoluteError","value":MeanAbsoluteError})
        dataforerrormetric.append({"Metric":"MeanSquaredError","value":MeanSquaredError})
        dataforerrormetric.append({"Metric":"RootMeanSquaredError","value":RootMeanSquaredError})
        dataforerrormetric.append({"Metric":"MeanSquaredLogError","value":MeanSquaredLogError})
        dataforerrormetric.append({"Metric":"MedianAbsoluteError","value":MedianAbsoluteError})
        
        print(featurenames)
        print(dataforcomparisonplot)
        # Metrics
        Explained_Variance_score=regressionmodel["Explained Variance Score"]
        print(Explained_Variance_score)
        return simplejson.dumps({"dataforcomparisonplot":dataforcomparisonplot,"dataforerrormetric":dataforerrormetric,"MedianAbsoluteError":MedianAbsoluteError,"MeanSquaredLogError":MeanSquaredLogError,"RootMeanSquaredError":RootMeanSquaredError,"MeanSquaredError":MeanSquaredError,"Ensemble": Ensemble,"MeanAbsoluteError":MeanAbsoluteError,"Type":Type ,"Estimators":Estimators ,"numfeatures":numfeatures ,"featurenames":featurenames ,"Variance_Threshold":Variance_Threshold ,"Univariate_selection":Univariate_selection ,"Explained_Variance_score":Explained_Variance_score},ignore_nan=True)
        
    return a


@app.route("/execute_classificationoperation", methods=['POST']) 
@cross_origin()
def executeclassification():
    if request.method=="POST":
        if request.files:
            print("ABCDEF")
        data = request.get_json(silent=True)
        item = {'values': data.get('values')}
        print(item)
        a="Successfully Submitted in Flask"
        mng_client2 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db2 = mng_client2['ampleai'] 
        collection_name2 = 'csv' 
        db_cm2 = mng_db2[collection_name2]
        data_from_db = db_cm2.find_one({"User":"Smit"})
        
        dictfromdatabase=data_from_db["data"]
        df2=pd.DataFrame.from_dict(dictfromdatabase)
        print(df2.head())
        # Our opeartion
        selectedtarget=item["values"]["select"]
        print(selectedtarget)
        df2[[selectedtarget]]
        try:
            df2.drop(["Unnamed: 0"],axis=1,inplace=True)
        except KeyError:
            pass
        X = df2.drop(selectedtarget,axis=1).values
        y = df2[selectedtarget].values
        Xfeature = df2.drop(selectedtarget,axis=1)
        yfeature = df2[selectedtarget]
        clf1 = LogisticRegression(random_state=1)
        clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf3 = GaussianNB()
        eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) 
        print(X.shape)
        sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
        sel.fit_transform(X)
        print(X.shape)
        accuracyscorelist=[]
        i=2
        lenofX=len(df2.drop(selectedtarget,axis=1).columns)
        while i<=lenofX:
            X_new = SelectKBest(score_func=chi2, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            eclass = eclf.fit(X_train, y_train)
            y_pred = eclass.predict(X_test)
            accuracyscorelist.append(balanced_accuracy_score(y_test, y_pred))
            i=i+1
        clf1accuracyscorelist=[]
        i=2
        while i<=lenofX:
            X_new = SelectKBest(score_func=chi2, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            trainedclf1 = clf1.fit(X_train, y_train)
            y_pred = trainedclf1.predict(X_test)
            clf1accuracyscorelist.append(balanced_accuracy_score(y_test, y_pred))
            i=i+1
        clf2accuracyscorelist=[]
        i=2
        while i<=lenofX:
            X_new = SelectKBest(score_func=chi2, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            trainedclf2 = clf2.fit(X_train, y_train)
            y_pred = trainedclf2.predict(X_test)
            clf2accuracyscorelist.append(balanced_accuracy_score(y_test, y_pred))
            i=i+1
        clf3accuracyscorelist=[]
        i=2
        while i<=lenofX:
            X_new = SelectKBest(score_func=chi2, k=i).fit_transform(X, y)
            X_new.shape
            X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
            trainedclf3 = clf3.fit(X_train, y_train)
            y_pred = trainedclf3.predict(X_test)
            clf3accuracyscorelist.append(balanced_accuracy_score(y_test, y_pred))
            i=i+1

        # Create data for comparison plot
        dataforcomparisonplot=[]
        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=clf1accuracyscorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"LogisticRegression"})
        
        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=clf2accuracyscorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"RandomForestClassifier"})
        
        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=clf3accuracyscorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"GaussianNB"})
        

        for featureindex in range(1,lenofX-1):
            valueofk=featureindex+1
            valueofscore=accuracyscorelist[featureindex]
            dataforcomparisonplot.append({"k":valueofk,"value":valueofscore,"category":"Ensemble"})
        
        print(dataforcomparisonplot)
        print(clf1accuracyscorelist)
        largestr2=max(accuracyscorelist)
        print(accuracyscorelist.index(largestr2))
        Optimalk=accuracyscorelist.index(largestr2) + 2
        print(Optimalk)
        # Create and fit selector
        selector = SelectKBest(score_func=chi2, k=Optimalk)
        selector.fit(X, y)
        # Get columns to keep and create new dataframe with those only
        cols = selector.get_support(indices=True)
        print(cols)
        Xcolumns=[]
        for col in Xfeature.columns:
            Xcolumns.append(col)
        print(Xcolumns)
        new_features = []
        for bool, feature in zip(cols, Xcolumns):
            if bool:
                new_features.append(feature)
        featurelist=new_features
        print(new_features)
        X_new = SelectKBest(score_func=chi2, k=Optimalk).fit_transform(X, y)
        print(X_new[0])
        print(X[0])
        firstrow=X[0].tolist()
        print(type(X[0]))
        
        
        # Predicting model 
        X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.2, random_state = 0)
        eclassf = eclf.fit(X_train, y_train)
        y_pred = eclassf.predict(X_test)
        
        # Make new Dictionary to sore in database
        classificationmodel={}
        classificationmodel["dataforcomparisonplot"]=dataforcomparisonplot
        classificationmodel["Ensemble"]="Yes"
        classificationmodel["Type"]="Voting Classifier"
        classificationmodel["Estimators"]=["LogisticRegression","RandomForestClassifier","GaussianNB"]
        classificationmodel["numfeatures"]=Optimalk
        classificationmodel["featurenames"]=featurelist
        classificationmodel["featureselection"]=["Variance Threshlod","Univariate Selection"]
        # Metrics
        classificationmodel["BalancedAccuracyModel"]=balanced_accuracy_score(y_test, y_pred)
        #classificationmodel["AveragePrecisionScore"]=average_precision_score(y_test, y_pred)
        classificationmodel["MacroF1Score"]=f1_score(y_test, y_pred, average='macro')
        classificationmodel["MicroF1Score"]=f1_score(y_test, y_pred, average='micro')
        classificationmodel["WeightedF1Score"]=f1_score(y_test, y_pred, average='weighted')
        classificationmodel["MacroRecallScore"]=recall_score(y_test, y_pred, average='macro')
        classificationmodel["MicroRecallScore"]=recall_score(y_test, y_pred, average='micro')
        classificationmodel["WeightedRecallScore"]=recall_score(y_test, y_pred, average='weighted')
        classificationmodel["JaccardScore"]=jaccard_score(y_test, y_pred, average='macro')
        #classificationmodel["ROCAUCScore"]=roc_auc_score(y_test, y_pred)
        # Create the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cmlist=cm.tolist()
        cmlength=len(cmlist)
        actuallist=[]
        for i in range(0,cmlength):
            strnum=str(i)
            element="Actual: " + strnum
            actuallist.append(element)
        predictionlist=[]
        for i in range(0,cmlength):
            strnum=str(i)
            element="Prediction: " + strnum
            predictionlist.append(element)
        dataformatrix=[]
        for i in range(1,cmlength+1):
            getlist=cmlist[i-1]
            count=0
            for listelement in getlist:
                if count>=2:
                    count=0
                strnum=str(i-1)
                actual="Actual"+strnum
                value=listelement
                strcount=str(count)
                predicted="Predicted"+str(count)
                count=count+1
                dataformatrix.append({"actual":actual,"value":value,"predicted":predicted})
        print(dataformatrix)
        classificationmodel["dataformatrix"]=dataformatrix
        output = []
        for x in y:
            if x not in output:
                output.append(x)
        noofclasses=len(output)
        classlist=[]
        for i in range(0,noofclasses):
            strnum=str(i)
            element="Class: " + strnum
            classlist.append(element)
        report=classification_report(y_test, y_pred, target_names=classlist)
        rep=" ".join(report.split())
        replist=list(rep.split())
        newreplist=[]
        for element in replist:
            try:
                newelement=float(element)
                newreplist.append(newelement)
            except ValueError:
                pass
        noofrows=len(newreplist)/4
        noofclasses=noofrows-3
        noofclasses=int(noofclasses)
        classlist=[]
        for i in range(0,noofclasses):
            strnum=str(i)
            element="Class: " + strnum
            classlist.append(element)
        classlist.append("Micro Avg")
        classlist.append("Macro Avg")
        classlist.append("Weighted Avg")
        Metricslist=["Precision","Recall","F1 Score","Support"]
        count=0
        dataforreport=[]
        for element in classlist:
            classvalue=element
            for metric in Metricslist:
                value=newreplist[count]
                count=count+1
                metricvalue=metric
                dataforreport.append({"value":value,"classvalue":classvalue,"metricvalue":metricvalue})
        
        classificationmodel["dataforreport"]=dataforreport
        
    
    
        # Store Updated dictionary in database
        db_cm2.remove()
        #
       
        #
        data_dict = df2.to_dict("records")
        s1=df2.to_csv(index=False)
        db_cm2.insert({"User":"Smit","data":data_dict,"s":s1,"classificationmodel":classificationmodel},check_keys=False)
    return a

@app.route("/fetch_classificationmodel", methods=['GET']) 
@cross_origin()
def fetchclassificationmodel():
    if request.method=="GET":
        mng_client1 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db1 = mng_client1['ampleai'] 
        collection_name1 = 'csv' 
        db_cm1 = mng_db1[collection_name1]
        data_from_db = db_cm1.find_one({"User":"Smit"})
        s=data_from_db["s"]
        df1=data_from_db["data"]
        classificationmodel=data_from_db["classificationmodel"]
        dataforcomparisonplot=classificationmodel["dataforcomparisonplot"]
        Ensemble=classificationmodel["Ensemble"]
        Type=classificationmodel["Type"]
        Estimators=classificationmodel["Estimators"]
        numfeatures=classificationmodel["numfeatures"]
        featurenames=classificationmodel["featurenames"]
        featureselection=classificationmodel["featureselection"]
        Variance_Threshold=featureselection[0]
        Univariate_selection=featureselection[1]
        # Metrics

        BalancedAccuracyModel=classificationmodel["BalancedAccuracyModel"]
        #AveragePrecisionScore=classificationmodel["AveragePrecisionScore"]
        MacroF1Score=classificationmodel["MacroF1Score"]
        MicroF1Score=classificationmodel["MicroF1Score"]
        WeightedF1Score=classificationmodel["WeightedF1Score"]
        MacroRecallScore=classificationmodel["MacroRecallScore"]
        MicroRecallScore=classificationmodel["MicroRecallScore"]
        WeightedRecallScore=classificationmodel["WeightedRecallScore"]
        JaccardScore=classificationmodel["JaccardScore"]
        dataforreport=classificationmodel["dataforreport"]
        dataformatrix=classificationmodel["dataformatrix"]
        #ROCAUCScore=classificationmodel["ROCAUCScore"]
        
        dataforscoremetric=[]
        #dataforscoremetric.append({"Metric":"AveragePrecisionScore","value":AveragePrecisionScore})
        dataforscoremetric.append({"Metric":"MacroF1Score","value":MacroF1Score})
        dataforscoremetric.append({"Metric":"MicroF1Score","value":MicroF1Score})
        dataforscoremetric.append({"Metric":"WeightedF1Score","value":WeightedF1Score})
        dataforscoremetric.append({"Metric":"MacroRecallScore","value":MacroRecallScore})
        dataforscoremetric.append({"Metric":"MicroRecallScore","value":MicroRecallScore})
        dataforscoremetric.append({"Metric":"WeightedRecallScore","value":WeightedRecallScore})
        dataforscoremetric.append({"Metric":"JaccardScore","value":JaccardScore})
        #dataforscoremetric.append({"Metric":"ROCAUCScore","value":ROCAUCScore})
        
        print(featurenames)
        print(dataforcomparisonplot)
        print(BalancedAccuracyModel)
        return simplejson.dumps({"dataforcomparisonplot":dataforcomparisonplot,"dataforreport":dataforreport,"dataformatrix":dataformatrix,"BalancedAccuracyModel":BalancedAccuracyModel,"dataforscoremetric":dataforscoremetric,"Ensemble": Ensemble,"Type":Type ,"Estimators":Estimators ,"numfeatures":numfeatures ,"featurenames":featurenames ,"Variance_Threshold":Variance_Threshold ,"Univariate_selection":Univariate_selection},ignore_nan=True)
        
    return a


@app.route("/execute_clusteringoperation", methods=['POST']) 
@cross_origin()
def executeclustering():
    if request.method=="POST":
        if request.files:
            print("ABCDEF")
        data = request.get_json(silent=True)
        item = {'values': data.get('values')}
        print(item)
        a="Successfully Submitted in Flask"
        mng_client2 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db2 = mng_client2['ampleai'] 
        collection_name2 = 'csv' 
        db_cm2 = mng_db2[collection_name2]
        data_from_db = db_cm2.find_one({"User":"Smit"})
        
        dictfromdatabase=data_from_db["data"]
        df2=pd.DataFrame.from_dict(dictfromdatabase)
        print(df2.head())
        ##################
        def tokenize_and_stem(text):
            stemmer = SnowballStemmer("english")
            # Tokenize by sentence, then by word
            tokens = [y for x in nltk.sent_tokenize(text) for y in nltk.word_tokenize(x)]
            
            # Filter out raw tokens to remove noise
            filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
            
            # Stem the filtered_tokens
            stems = [stemmer.stem(word) for word in filtered_tokens]
            
            return stems
        # Our opeartion
        clusteringmodel={}
        selectedtarget=item["values"]["selecttype"]
        print(selectedtarget)
        if selectedtarget=="Numeric":
            selectedcolumns=item["values"]["select-multiple"]
            datm=df2[selectedcolumns]
            from sklearn.preprocessing import StandardScaler
            standard_scaler = StandardScaler()
            dat2 = standard_scaler.fit_transform(datm)
            from sklearn.decomposition import PCA
            pca = PCA(svd_solver='randomized', random_state=42)
            #let's apply PCA
            pca.fit(dat2)
            #List of PCA components.It would be the same as the number of variables
            pca.components_
            #Let's check the variance ratios
            arraypca=pca.explained_variance_ratio_
            listpca=arraypca.tolist()
            sumstart=0
            sum=0
            elementid=0
            pcalist=[]
            for element in listpca:
                sumstart=element
                sum=sum+sumstart
                if sum>=0.9:
                    pcalist.append(elementid)
                elementid=elementid+1
            numpca=pcalist[0]
            from sklearn.decomposition import IncrementalPCA
            pca_final = IncrementalPCA(n_components=numpca)
            df_train_pca = pca_final.fit_transform(dat2)
            df_train_pca.shape
            #take the transpose of the PC matrix so that we can create the new matrix
            pc = np.transpose(df_train_pca)
            pcadict={}
            for i in range(1,numpca+1):
                dictid="PCA"+str(i)
                dictvalue=pc[i-1]
                pcadict[dictid]=dictvalue
            pcs_df2 = pd.DataFrame(pcadict)
            dat3 = pcs_df2
            dat3_1 = standard_scaler.fit_transform(dat3)
            ##################
            def optimalK(data,initial=1, nrefs=3, maxClusters=11):
                """
                Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
                Params:
                    data: ndarry of shape (n_samples, n_features)
                    nrefs: number of sample reference datasets to create
                    maxClusters: Maximum number of clusters to test for
                Returns: (gaps, optimalK)
                """
                gaps = np.zeros((len(range(initial, maxClusters)),))
                resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
                for gap_index, k in enumerate(range(initial, maxClusters)):

                    # Holder for reference dispersion results
                    refDisps = np.zeros(nrefs)

                    # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
                    for i in range(nrefs):
                        
                        # Create new random reference set
                        randomReference = np.random.random_sample(size=data.shape)
                        
                        # Fit to it
                        km = KMeans(k)
                        km.fit(randomReference)
                        
                        refDisp = km.inertia_
                        refDisps[i] = refDisp

                    # Fit cluster to original data and create dispersion
                    km = KMeans(k)
                    km.fit(data)
                    
                    origDisp = km.inertia_

                    # Calculate gap statistic
                    gap = np.log(np.mean(refDisps)) - np.log(origDisp)

                    # Assign this loop's gap statistic to gaps
                    gaps[gap_index] = gap
                    resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

                return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal



            ################            
            maxclus=11
            initial=1
            gaplist=[]
            k, gapdf = optimalK(dat3_1,initial=initial, nrefs=5, maxClusters=maxclus)
            print ('Optimal k is: ', k)
            print(gapdf.clusterCount)
            print(gapdf.gap)
            gaplist.extend(gapdf.gap.tolist())
            print(gapdf.gap)
            while k>=int(10*0.8):
                maxclus=maxclus+10
                print("INSIDE")
                initial=initial+10
                k, gapdf = optimalK(dat3_1,initial=initial, nrefs=5, maxClusters=maxclus)
                print ('Optimal k is: ', k)
                print(gapdf.clusterCount)
                print(gapdf.gap)
                convlist=gapdf.gap.tolist()
                print(convlist)
                gaplist.extend(convlist)
                if len(gaplist)>90:
                    k=1



            print(gaplist) 
            print(max(gaplist))
            maxlist=max(gaplist)
            gapoptimalk=gaplist.index(maxlist)
            print(gapoptimalk)
            dataforgap=[]
            gapelecount=1
            for element in gaplist:
                dataforgap.append({"Xaxis":gapelecount,"Yaxis":element})
                gapelecount=gapelecount+1

            sil_score_max = -1 #this is the minimum possible score
            dataforsilhouette=[]
            for n_clusters in range(2,maxclus):
                kmsil  = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init=1)
                labels = kmsil .fit_predict(dat3_1)
                sil_score = silhouette_score(dat3_1, labels)
                print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
                dataforsilhouette.append({"Xaxis":n_clusters,"Yaxis":sil_score})
                if sil_score > sil_score_max:
                    sil_score_max = sil_score
                    best_n_clusters = n_clusters
            print(best_n_clusters)

            km = KMeans(n_clusters=best_n_clusters)
            km.fit(dat3_1)
            clusters = km.labels_.tolist()
            df3=df2
            df3["cluster"] = clusters
            ######################

            columndict={}
            reactscatteriter=[]
            clusterlist=df3["cluster"].tolist()
            count=0
            for element in selectedcolumns:
                scatterplotdict={}
                scatterplotdict["columnname"]=element
                xaxislist=df3[element].tolist()
                Xaxis=xaxislist
                dupintcolumns=selectedcolumns.copy()
                dupintcolumns.remove(element)
                scatterplotdict["scatteriter"]=[]
                for subelement in dupintcolumns:
                    yaxislist=df3[subelement].tolist()
                    liscount=0
                    scatterlist=[]
                    for rangeele in range(0,len(xaxislist)):
                        Xvalue=xaxislist[rangeele]
                        Yvalue=yaxislist[rangeele]
                        Cluster=clusterlist[rangeele]
                        scatterlist.append({"Xvalue":Xvalue,"YValue":Yvalue,"Cluster":Cluster})
                    scatterplotdict["scatteriter"].append({"data":scatterlist,"xField":"Xvalue" ,"Xaxis":element,"Yaxis":subelement,
                                               "yField": 'Yvalue',"colorField": 'Cluster',
                                               "pointStyle": {"fillOpacity": 1, }, "xAxis": {"visible": "true","min": -5,},})
                reactscatteriter.append(scatterplotdict)
                count=count+1
        
     
    
            ######################
            clusteringmodel["featurename"]=[]
            clusteringmodel["featurename"].extend(selectedcolumns)
            clusteringmodel["numoffeatures"]=len(selectedcolumns)
            clusteringmodel["dataprocessing"]="Principal Component Analysis"
            clusteringmodel["clusterselection"]="Silhouette Score"
            clusteringmodel["reactscatteriter"]=reactscatteriter
        elif selectedtarget=="Character":
            selectedcolumn=item["values"]["select"]
            tfidf_vectorizer = TfidfVectorizer(max_df=0.95, max_features=200000, min_df=0.05, stop_words='english', use_idf=True, tokenizer=tokenize_and_stem,ngram_range=(1,3))
            tfidf_matrix = tfidf_vectorizer.fit_transform([str(x) for x in df2[selectedcolumn]])
            print(tfidf_vectorizer.get_feature_names())

            print(tfidf_matrix.shape)
            print(tfidf_matrix)
            ##################
            def optimalK(data,initial=1, nrefs=3, maxClusters=15):
                """
                Calculates KMeans optimal K using Gap Statistic from Tibshirani, Walther, Hastie
                Params:
                    data: ndarry of shape (n_samples, n_features)
                    nrefs: number of sample reference datasets to create
                    maxClusters: Maximum number of clusters to test for
                Returns: (gaps, optimalK)
                """
                gaps = np.zeros((len(range(initial, maxClusters)),))
                resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
                for gap_index, k in enumerate(range(initial, maxClusters)):

                    # Holder for reference dispersion results
                    refDisps = np.zeros(nrefs)

                    # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
                    for i in range(nrefs):
                        
                        # Create new random reference set
                        randomReference = np.random.random_sample(size=data.shape)
                        
                        # Fit to it
                        km = KMeans(k)
                        km.fit(randomReference)
                        
                        refDisp = km.inertia_
                        refDisps[i] = refDisp

                    # Fit cluster to original data and create dispersion
                    km = KMeans(k)
                    km.fit(data)
                    
                    origDisp = km.inertia_

                    # Calculate gap statistic
                    gap = np.log(np.mean(refDisps)) - np.log(origDisp)

                    # Assign this loop's gap statistic to gaps
                    gaps[gap_index] = gap
                    resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)

                return (gaps.argmax() + 1, resultsdf)  # Plus 1 because index of 0 means 1 cluster is optimal, index 2 = 3 clusters are optimal



            ################            
            maxclus=15
            initial=1
            gaplist=[]
            k, gapdf = optimalK(tfidf_matrix,initial=initial, nrefs=5, maxClusters=maxclus)
            print ('Optimal k is: ', k)
            print(gapdf.clusterCount)
            print(gapdf.gap)
            gaplist.extend(gapdf.gap.tolist())
            print(gapdf.gap)
            while k>=int(14*0.8):
                maxclus=maxclus+14
                print("INSIDE")
                initial=initial+14
                k, gapdf = optimalK(tfidf_matrix,initial=initial, nrefs=5, maxClusters=maxclus)
                print ('Optimal k is: ', k)
                print(gapdf.clusterCount)
                print(gapdf.gap)
                convlist=gapdf.gap.tolist()
                print(convlist)
                gaplist.extend(convlist)
                if len(gaplist)>98:
                    k=1



            print(gaplist) 
            print(max(gaplist))
            maxlist=max(gaplist)
            gapoptimalk=gaplist.index(maxlist)
            print(gapoptimalk)
            dataforgap=[]
            gapelecount=1
            for element in gaplist:
                dataforgap.append({"Xaxis":gapelecount,"Yaxis":element})
                gapelecount=gapelecount+1

            sil_score_max = -1 #this is the minimum possible score
            dataforsilhouette=[]
            for n_clusters in range(2,maxclus):
                kmsil  = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init=1)
                labels = kmsil .fit_predict(tfidf_matrix)
                sil_score = silhouette_score(tfidf_matrix, labels)
                print("The average silhouette score for %i clusters is %0.2f" %(n_clusters,sil_score))
                dataforsilhouette.append({"Xaxis":n_clusters,"Yaxis":sil_score})
                if sil_score > sil_score_max:
                    sil_score_max = sil_score
                    best_n_clusters = n_clusters
            print(best_n_clusters)

            km = KMeans(n_clusters=gapoptimalk)
            km.fit(tfidf_matrix)
            clusters = km.labels_.tolist()
            df3=df2
            df3["cluster"] = clusters
            ######################
            clusteringmodel["featurename"]=[]
            clusteringmodel["featurename"].append(selectedcolumn)
            clusteringmodel["numoffeatures"]=1
            clusteringmodel["dataprocessing"]="TF-IDF Vectorizer"
            clusteringmodel["clusterselection"]="Gap Statistic"
        clusteringmodel["dataforgap"]=dataforgap
        clusteringmodel["dataforsilhouette"]=dataforsilhouette
        clusteringmodel["numofclusters"]=best_n_clusters
        
        
        # Store Updated dictionary in database
        db_cm2.remove()
        #
       
        #
        data_dict = df2.to_dict("records")
        s1=df2.to_csv(index=False)
        db_cm2.insert({"User":"Smit","data":data_dict,"s":s1,"clusteringmodel":clusteringmodel},check_keys=False)
        return "PYTHON METHOD EXECUTED"



@app.route("/fetch_clusteringmodel", methods=['GET']) 
@cross_origin()
def fetchclusteringmodel():
    if request.method=="GET":
        mng_client1 = pymongo.MongoClient("mongodb+srv://expertron:smitenter@cluster0-dpwn4.mongodb.net/test?retryWrites=true&w=majority")
        mng_db1 = mng_client1['ampleai'] 
        collection_name1 = 'csv' 
        db_cm1 = mng_db1[collection_name1]
        data_from_db = db_cm1.find_one({"User":"Smit"})
        s=data_from_db["s"]
        df1=data_from_db["data"]
        clusteringmodel=data_from_db["clusteringmodel"]
        featurename=clusteringmodel["featurename"]
        numoffeatures=clusteringmodel["numoffeatures"]
        dataforgap=clusteringmodel["dataforgap"]
        dataforsilhouette=clusteringmodel["dataforsilhouette"]
        numofclusters=clusteringmodel["numofclusters"]
        dataprocessing=clusteringmodel["dataprocessing"]
        reactscatteriter=clusteringmodel["reactscatteriter"]
       
        
        
        return simplejson.dumps({"featurename":featurename,"reactscatteriter":reactscatteriter,"numoffeatures":numoffeatures,"dataforgap":dataforgap,"dataforsilhouette":dataforsilhouette,"numofclusters":numofclusters,"dataprocessing":dataprocessing},ignore_nan=True)
        
    return a


if __name__ == "__main__":
    
    app.run()
