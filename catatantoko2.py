from flask import Flask, render_template, request
from flask_mysqldb import MySQL
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'toko'
mysql = MySQL(app)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        details = request.form
        Pendapatan = details['dapat']
        Biayaiklan = details['iklan']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO keuangan (pendapatan, biayaiklan) VALUES (%s, %s)", (Pendapatan, Biayaiklan))
        mysql.connection.commit()
        cur.close()
        return 'success'
    return render_template('index.html')
@app.route('/untung')
def untung():
 cur = mysql.connection.cursor()
 cur.execute('''SELECT pendapatan, biayaiklan FROM keuangan''')
 rv = cur.fetchall()
 return render_template("tabelpendapatan.html",value=rv)
@app.route('/buatfile')
def buatfile():
 cur = mysql.connection.cursor()
 cur.execute('''SELECT biayaiklan, pendapatan FROM keuangan''')
 rv = cur.fetchall()
 c = csv.writer(open('datatoko.csv','w'))
 for x in rv:
  c.writerow(x)
 return 'sukses'
@app.route('/model')
def modelml():
 dataset=pd.read_csv('datatoko.csv')
 X=dataset.iloc[:,:-1].values
 y=dataset.iloc[:,1].values
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state=0)
 regressor = LinearRegression()
 regressor.fit(X_train, y_train)
 y_pred = regressor.predict(X_test)
 pickle.dump(regressor, open('model.pkl','wb'))
 model = pickle.load(open('model.pkl','rb'))
 result=float(model.predict([[75]]))
 return render_template("hasil.html", result=result)
if __name__ == '__main__':
    app.run()