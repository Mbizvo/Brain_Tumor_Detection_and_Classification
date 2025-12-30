from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
import os
import cv2
import numpy as np
from keras.models import load_model
from keras.metrics import AUC
from sqlalchemy import func, extract

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///new_site.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

db = SQLAlchemy(app)
migrate = Migrate(app, db)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

dependencies = {
    'auc_roc': AUC
}

categories = ["Glioma_Tumor", "Meningioma_Tumor", "No_Tumor", "Pituitary_Tumor"]

# Load the model
model = load_model('Brain_tumor_detection_res50.h5', custom_objects=dependencies)
model.make_predict_function()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    title = db.Column(db.String(100), nullable=True)  # Make title nullable

class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    tumor_type = db.Column(db.String(100), nullable=False)
    date_diagnosed = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), unique=True, nullable=False)
    prediction = db.Column(db.String(120), nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Unable to load image from path:", img_path)
        return None

    img = cv2.resize(img, (200, 200))
    img = np.array(img, dtype=np.float32) / 255.0
    prediction = model.predict(np.expand_dims(img, axis=0))
    predicted_class = categories[np.argmax(prediction)]
    return predicted_class

@app.route("/")
@app.route("/about")
def about():
    return render_template('about.html')

@app.route("/first")
def first():
    return render_template('first.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        title = request.form.get('title', 'Default Title')  # Provide a default value if title is not provided
        
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already taken. Please choose a different username.')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password=hashed_password, title=title)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['title'] = user.title
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.')
    return render_template('login.html')

@app.route("/logout")
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('title', None)
    return redirect(url_for('login'))

@app.before_request
def require_login():
    if 'user_id' not in session and request.endpoint not in ['login', 'register', 'about', 'first', 'static']:
        return redirect(url_for('login'))

@app.route("/index", methods=['GET', 'POST'])
def index():
    return render_template("index.html")

@app.route("/add_patient", methods=['GET', 'POST'])
def add_patient():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        tumor_type = request.form['tumor_type']
        date_diagnosed = request.form['date_diagnosed']
        
        if not age.isdigit() or int(age) <= 0:
            flash('Age must be a positive number.')
            return redirect(url_for('add_patient'))
        
        new_patient = Patient(
            name=name, 
            age=int(age), 
            gender=gender, 
            tumor_type=tumor_type, 
            date_diagnosed=datetime.strptime(date_diagnosed, '%Y-%m-%d')
        )
        db.session.add(new_patient)
        db.session.commit()
        flash('Patient added successfully!')
        return redirect(url_for('index'))
    return render_template('add_patient.html')

@app.route("/submit", methods=['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']
        if img and allowed_file(img.filename):
            filename = secure_filename(img.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure the upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            img.save(img_path)

            predict_result = predict_image(img_path)

            # Check if a prediction with the same filename already exists
            existing_prediction = Prediction.query.filter_by(filename=filename).first()
            if existing_prediction:
                existing_prediction.prediction = predict_result
            else:
                new_prediction = Prediction(filename=filename, prediction=predict_result)
                db.session.add(new_prediction)
            
            db.session.commit()

            return render_template("prediction.html", prediction=predict_result, img_path=img_path)
        else:
            flash('Invalid file type. Please upload an image file.')
            return redirect(request.url)
    return render_template("index.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/performance")
def performance():
    return render_template('performance.html')

@app.route("/chart")
def chart():
    return render_template('chart.html')

@app.route('/report')
def report():
    # Query all patient records
    reports = Patient.query.all()

    # Initialize counters for statistics
    total_tumors = len(reports)
    male_count = sum(1 for r in reports if r.gender.lower() == 'male')
    female_count = sum(1 for r in reports if r.gender.lower() == 'female')
    meningioma_count = sum(1 for r in reports if r.tumor_type.lower() == 'meningioma')
    glioma_count = sum(1 for r in reports if r.tumor_type.lower() == 'glioma')
    pituitary_count = sum(1 for r in reports if r.tumor_type.lower() == 'pituitary')
    no_tumor_count = sum(1 for r in reports if r.tumor_type.lower() == 'no tumor')
    age_above_40 = sum(1 for r in reports if r.age >= 40)
    age_below_40 = sum(1 for r in reports if r.age < 40)

    # Calculate tumors diagnosed per month
    tumors_per_month = {}
    for month in range(1, 13):
        count = Patient.query.filter(extract('month', Patient.date_diagnosed) == month).count()
        tumors_per_month[datetime(2000, month, 1).strftime('%B')] = count

    # Calculate tumors diagnosed per year from 2023 onwards
    tumors_per_year = {}
    start_year = 2023
    current_year = datetime.now().year
    for year in range(start_year, current_year + 1):
        count = Patient.query.filter(extract('year', Patient.date_diagnosed) == year).count()
        tumors_per_year[year] = count

    return render_template('report.html', reports=reports, total_tumors=total_tumors,
                           male_count=male_count, female_count=female_count,
                           meningioma_count=meningioma_count, glioma_count=glioma_count,
                           pituitary_count=pituitary_count, no_tumor_count=no_tumor_count,
                           age_above_40=age_above_40, age_below_40=age_below_40,
                           tumors_per_month=tumors_per_month, tumors_per_year=tumors_per_year)

@app.route('/delete_patient/<int:patient_id>', methods=['POST'])
def delete_patient(patient_id):
    patient = Patient.query.get(patient_id)
    if patient:
        db.session.delete(patient)
        db.session.commit()
        flash(f'Patient {patient.name} deleted successfully!', 'success')
    else:
        flash('Patient not found.', 'danger')
    return redirect(url_for('report'))

if __name__ == '__main__':
    app.run(debug=True)






