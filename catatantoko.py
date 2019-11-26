from flask import Flask, render_template, request
from flask_mysqldb import MySQL
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
if __name__ == '__main__':
    app.run()