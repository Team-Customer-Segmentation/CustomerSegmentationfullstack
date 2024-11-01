import sqlite3
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from sklearn.metrics import silhouette_score, davies_bouldin_score
import random

app = Flask(__name__, static_folder='static')
app.secret_key = 'test@123'

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    if request.method =='POST':
        pass
    return render_template('upload.html') 

@app.route('/Predict', methods=['GET', 'POST'])
def predict_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        
        file = request.files['file']

        if file.filename == '':
            return "No file Selected"

        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)

        new_data = pd.read_csv(file_path)

        required_features = ['Total_Spend', 'Recency', 'Total_No_Invoice_Generated']

        if not all(feature in new_data.columns for feature in required_features):
            return "Dataset missing few features"

        new_data = new_data[required_features]

        scaler = StandardScaler()
        scaler_features = scaler.fit_transform(new_data)

        kmeans = joblib.load('kmeans_model.pkl')
        new_data['Cluster'] = kmeans.predict(scaler_features)

        silhouette_avg = silhouette_score(scaler_features, new_data['Cluster'])
        
        cluster_scores = new_data.groupby('Cluster').apply(
            lambda x: (x['Total_Spend'].mean() + 
                        x['Total_No_Invoice_Generated'].mean() - 
                        x['Recency'].mean())
        )
        
        best_cluster = cluster_scores.idxmax()  
        best_cluster_data = new_data[new_data['Cluster'] == best_cluster]

        result_html = best_cluster_data.to_html()  

        return render_template('result.html', 
                               result_html=result_html,
                               silhouette_avg=silhouette_avg)


def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_users_table():
    conn = get_db_connection()
    conn.execute('''
                 CREATE TABLE IF NOT EXISTS users(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password TEXT NOT NULL
                 )
                 ''')
    conn.commit()
    conn.close()

create_users_table()

@app.route('/')
def home():
    return render_template('Home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        re_password = request.form['re_password']

        if password != re_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        try:
            conn.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', 
                         (name, email, hashed_password))
            conn.commit()
            flash('Registration successful! Please log in.')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists. Please choose another one.')
            return redirect(url_for('register'))
        finally:
            conn.close()

    return render_template('RegisterPage.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = user['name']
            flash('Login successful!')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.')
            return redirect(url_for('login'))

    return render_template('Login_page.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('You need to log in first.')
        return redirect(url_for('login'))
    
    df = pd.read_csv('best_cluster.csv')
    data = df.to_dict(orient='records')
    column_names = df.columns.values

    return render_template('Dashboard.html', username=session['username'], column_names=column_names, data=data)

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

@app.route('/show_users')
def show_users():
    conn = get_db_connection()
    users = conn.execute('SELECT * FROM users').fetchall()
    conn.close()
    return str(users) 


if __name__ == '__main__':
    app.run(debug=True)
