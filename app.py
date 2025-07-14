from flask import Flask, render_template, request, jsonify, session
import os
import face_recognition
import cv2
import dlib
import numpy as np
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify
import sqlite3
import shutil
import glob
import threading

app = Flask(__name__)
app.secret_key = '242nm3b2khv3h2'  # Set a strong, unique secret key for session management

DATASET_DIR = 'dataset'
ENCODINGS_FILE = os.path.join(DATASET_DIR, 'encodings.json')
ATTENDANCE_FILE = 'attendance.json'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')

# Create directories if they don't exist
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Initialize known face encodings and names
known_face_encodings = []
known_face_names = []

# --- SQLite Setup ---
DB_FILE = 'attendance.db'

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def load_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings = []
    known_face_names = []
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'r') as f:
            data = json.load(f)
            for entry in data:
                known_face_encodings.append(np.array(entry['encoding']))
                known_face_names.append(entry['name'])
    print(f"Loaded {len(known_face_names)} known faces.")

load_known_faces()

@app.route('/')
def index():
    return render_template('index.html')

# Utility: Always use HOG model for fast face detection
def get_face_locations(img):
    return face_recognition.face_locations(img, model="hog")

@app.route('/register_face', methods=['POST'])
def register_face():
    try:
        data = request.get_json()
        image_data = data['image']
        name = data['name']

        # Decode base64 image
        encoded_data = image_data.split(',')[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert BGR to RGB (face_recognition uses RGB)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Use stricter model and tolerance
        face_locations = get_face_locations(rgb_img)
        if len(face_locations) == 0:
            return jsonify({'status': 'error', 'message': 'No face detected in the image. Please try again.'})
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        if len(face_encodings) > 0:
            # Allow multiple encodings per name
            for face_encoding in face_encodings:
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            # Save to JSON file
            encodings_to_save = []
            for i, encoding in enumerate(known_face_encodings):
                encodings_to_save.append({
                    'name': known_face_names[i],
                    'encoding': encoding.tolist()
                })
            with open(ENCODINGS_FILE, 'w') as f:
                json.dump(encodings_to_save, f)
            load_known_faces()  # Refresh in-memory lists
            # Save image
            cv2.imwrite(os.path.join(IMAGES_DIR, f'{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'), img)
            return jsonify({'status': 'success', 'message': f'Face for {name} registered successfully!'})
        else:
            return jsonify({'status': 'error', 'message': 'No face detected in the image. Please try again.'})
    except Exception as e:
        print(f"Error in register_face: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred during face registration. Please try again.'})

import smtplib
from email.mime.text import MIMEText
from threading import Thread
import time

# Configuration for attendance scheduling and email
SCHEDULE_CONFIG_FILE = 'schedule_config.json'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'your_email@gmail.com'  # TODO: Replace with your email
SENDER_PASSWORD = 'your_email_password'  # TODO: Replace with your email password or app password

# Global variables to store the scheduled window
scheduled_attendance_start_time = None
scheduled_attendance_end_time = None

def load_schedule_config():
    global scheduled_attendance_start_time, scheduled_attendance_end_time
    if os.path.exists(SCHEDULE_CONFIG_FILE):
        with open(SCHEDULE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            scheduled_attendance_start_time = config.get('attendance_start_time')
            scheduled_attendance_end_time = config.get('attendance_end_time')
    print(f"Loaded scheduled attendance window: {scheduled_attendance_start_time} to {scheduled_attendance_end_time}")

load_schedule_config()

# --- Save attendance to SQLite ---
def save_attendance_record(name, status):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Upsert: if record exists for this name and date, update it; else insert
    c.execute('SELECT id FROM attendance WHERE name = ? AND date = ?', (name, date_str))
    row = c.fetchone()
    if row:
        c.execute('UPDATE attendance SET time = ?, status = ? WHERE id = ?', (time_str, status, row[0]))
    else:
        c.execute('INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)', (name, date_str, time_str, status))
    conn.commit()
    conn.close()

# --- Query attendance from SQLite ---
def query_attendance_records(date=None, name=None):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    query = 'SELECT id, name, date, time, status FROM attendance WHERE 1=1'
    params = []
    if date:
        query += ' AND date = ?'
        params.append(date)
    if name:
        query += ' AND name LIKE ?'
        params.append(f'%{name}%')
    query += ' ORDER BY date DESC, time DESC'
    c.execute(query, params)
    rows = c.fetchall()
    conn.close()
    records = [
        {'id': row[0], 'name': row[1], 'date': row[2], 'time': row[3], 'status': row[4]}
        for row in rows
    ]
    return records

def send_absent_notification(absent_students):
    if not absent_students:
        print("No absent students to notify.")
        return

    # TODO: Replace with actual recipient emails or implement SMS API
    recipient_email = 'recipient_email@example.com' 

    subject = "Attendance Notification: Absent Today"
    body = f"The following students were marked absent today: {', '.join(absent_students)}."

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"Absent notification sent to {recipient_email} for: {', '.join(absent_students)}")
    except Exception as e:
        print(f"Failed to send email notification: {e}")

@app.route('/attendance_table')
def attendance_table():
    return render_template('attendance.html')

@app.route('/get_attendance')
def get_attendance():
    date = request.args.get('date')
    name = request.args.get('name')
    records = query_attendance_records(date=date, name=name)
    return jsonify({'status': 'success', 'records': records})

@app.route('/set_schedule', methods=['POST'])
def set_schedule():
    global scheduled_attendance_start_time, scheduled_attendance_end_time
    data = request.get_json()
    start_time = data.get('start_time') # Expected format: HH:MM
    end_time = data.get('end_time')     # Expected format: HH:MM
    if start_time and end_time:
        try:
            datetime.strptime(start_time, '%H:%M')
            datetime.strptime(end_time, '%H:%M')
            scheduled_attendance_start_time = start_time
            scheduled_attendance_end_time = end_time
            with open(SCHEDULE_CONFIG_FILE, 'w') as f:
                json.dump({'attendance_start_time': scheduled_attendance_start_time, 'attendance_end_time': scheduled_attendance_end_time}, f)
            return jsonify({'status': 'success', 'message': f'Attendance scheduled from {scheduled_attendance_start_time} to {scheduled_attendance_end_time}'})
        except ValueError:
            return jsonify({'status': 'error', 'message': 'Invalid time format. Please use HH:MM.'})
    return jsonify({'status': 'error', 'message': 'Start and end time required.'})

@app.route('/get_schedule', methods=['GET'])
def get_schedule():
    return jsonify({'status': 'success', 'scheduled_start_time': scheduled_attendance_start_time, 'scheduled_end_time': scheduled_attendance_end_time})

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    try:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data provided.'})
        image_data = data['image'].split(',')[1]  # Remove 'data:image/jpeg;base64,' prefix
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({'status': 'error', 'message': 'Could not decode image.'})
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = get_face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        recognized_names = []
        for face_encoding in face_encodings:
            if len(known_face_encodings) == 0:
                continue
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
            name = "Unknown"
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.45:
                name = known_face_names[best_match_index]
            if name != "Unknown":
                recognized_names.append(name)
                today_date = datetime.now().strftime('%Y-%m-%d')
                records_today = query_attendance_records(date=today_date, name=name)
                already_marked_today = any(
                    rec['name'] == name and rec['status'] == 'Present'
                    for rec in records_today
                )
                if not already_marked_today:
                    save_attendance_record(name, 'Present')
        if recognized_names:
            return jsonify({'status': 'success', 'message': f'Attendance marked for {', '.join(set(recognized_names))}.'})
        elif face_locations:
            return jsonify({'status': 'info', 'message': 'Faces detected, but no registered faces recognized.'})
        else:
            return jsonify({'status': 'info', 'message': 'No faces detected in the frame.'})
    except Exception as e:
        print(f"Error in take_attendance: {e}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'})

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/bulk_register_faces', methods=['POST'])
def bulk_register_faces():
    try:
        data = request.get_json()
        image_data = data['image']
        name = data['name']
        encoded_data = image_data.split(',')[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = get_face_locations(rgb_img)
        if len(face_locations) == 0:
            return jsonify({'status': 'error', 'message': 'No face detected in the image.'})
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if len(face_encodings) > 0:
            for face_encoding in face_encodings:
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            encodings_to_save = []
            for i, encoding in enumerate(known_face_encodings):
                encodings_to_save.append({
                    'name': known_face_names[i],
                    'encoding': encoding.tolist()
                })
            with open(ENCODINGS_FILE, 'w') as f:
                json.dump(encodings_to_save, f)
            load_known_faces()
            cv2.imwrite(os.path.join(IMAGES_DIR, f'{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'), img)
            return jsonify({'status': 'success', 'message': f'Face for {name} registered.'})
        else:
            return jsonify({'status': 'error', 'message': 'No face encoding found.'})
    except Exception as e:
        print(f"Error in bulk_register_faces: {e}")
        return jsonify({'status': 'error', 'message': 'An error occurred during bulk registration.'})

@app.route('/get_roster')
def get_roster():
    students = []
    # Load from encodings.json and match to images
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'r') as f:
            data = json.load(f)
            for entry in data:
                name = entry['name']
                # Assume image is stored as dataset/images/{name}.jpg (or .png)
                # Try jpg, then png
                for ext in ['jpg', 'jpeg', 'png']:
                    image_path = os.path.join(IMAGES_DIR, f'{name}.{ext}')
                    if os.path.exists(image_path):
                        image_url = f'/images/{name}.{ext}'
                        break
                else:
                    image_url = ''
                students.append({'name': name, 'image_url': image_url})
    return jsonify({'status': 'success', 'students': students})

# Serve images statically
from flask import send_from_directory
@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/bulk_register')
def bulk_register():
    return render_template('bulk_register.html')

@app.route('/roster')
def roster():
    return render_template('roster.html')

@app.route('/delete_student', methods=['POST'])
def delete_student():
    data = request.get_json()
    name = data.get('name')
    if not name:
        return jsonify({'status': 'error', 'message': 'No name provided.'})
    # Remove from encodings
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'r') as f:
            encodings = json.load(f)
        encodings = [e for e in encodings if e['name'] != name]
        with open(ENCODINGS_FILE, 'w') as f:
            json.dump(encodings, f)
        load_known_faces()
    # Remove image(s)
    for ext in ['jpg', 'jpeg', 'png']:
        img_path = os.path.join(IMAGES_DIR, f'{name}.{ext}')
        if os.path.exists(img_path):
            os.remove(img_path)
    return jsonify({'status': 'success', 'message': f'Student {name} deleted.'})

@app.route('/update_student_name', methods=['POST'])
def update_student_name():
    data = request.get_json()
    old_name = data.get('old_name')
    new_name = data.get('new_name')
    if not old_name or not new_name:
        return jsonify({'status': 'error', 'message': 'Both old and new names required.'})
    # Update encodings
    updated = False
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'r') as f:
            encodings = json.load(f)
        for e in encodings:
            if e['name'] == old_name:
                e['name'] = new_name
                updated = True
        with open(ENCODINGS_FILE, 'w') as f:
            json.dump(encodings, f)
        load_known_faces()
    # Rename image file(s)
    for ext in ['jpg', 'jpeg', 'png']:
        old_path = os.path.join(IMAGES_DIR, f'{old_name}.{ext}')
        new_path = os.path.join(IMAGES_DIR, f'{new_name}.{ext}')
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
    if updated:
        return jsonify({'status': 'success', 'message': f'Name updated to {new_name}.'})
    else:
        return jsonify({'status': 'error', 'message': 'Student not found.'})

@app.route('/update_student_image', methods=['POST'])
def update_student_image():
    data = request.get_json()
    name = data.get('name')
    image_data = data.get('image')
    if not name or not image_data:
        return jsonify({'status': 'error', 'message': 'Name and image required.'})
    try:
        encoded_data = image_data.split(',')[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        cv2.imwrite(os.path.join(IMAGES_DIR, f'{name}_{datetime.now().strftime("%Y%m%d%H%M%S")}.jpg'), img)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = get_face_locations(rgb_img)
        if len(face_locations) == 0:
            return jsonify({'status': 'error', 'message': 'No face detected in the image.'})
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            if os.path.exists(ENCODINGS_FILE):
                with open(ENCODINGS_FILE, 'r') as f:
                    encodings = json.load(f)
                for e in encodings:
                    if e['name'] == name:
                        e['encoding'] = face_encoding.tolist()
                with open(ENCODINGS_FILE, 'w') as f:
                    json.dump(encodings, f)
                load_known_faces()
            return jsonify({'status': 'success', 'message': 'Image and encoding updated.'})
        else:
            return jsonify({'status': 'error', 'message': 'No face encoding found.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error updating image: {e}'})

@app.route('/attendance_dates')
def attendance_dates():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT DISTINCT date FROM attendance ORDER BY date DESC')
    dates = [row[0] for row in c.fetchall()]
    conn.close()
    return jsonify({'status': 'success', 'dates': dates})

@app.route('/attendance_for_date')
def attendance_for_date():
    date = request.args.get('date')
    if not date:
        return jsonify({'status': 'error', 'message': 'No date provided.'})
    records = query_attendance_records(date=date)
    return jsonify({'status': 'success', 'records': records})

@app.route('/update_attendance_status', methods=['POST'])
def update_attendance_status():
    data = request.get_json()
    date = data.get('date')
    name = data.get('name')
    status = data.get('status')
    if not date or not name or not status:
        return jsonify({'status': 'error', 'message': 'Missing fields.'})
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('UPDATE attendance SET status = ? WHERE date = ? AND name = ?', (status, date, name))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'Attendance updated.'})

@app.route('/delete_attendance_day', methods=['POST'])
def delete_attendance_day():
    data = request.get_json()
    date = data.get('date')
    if not date:
        return jsonify({'status': 'error', 'message': 'No date provided.'})
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM attendance WHERE date = ?', (date,))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': f'Attendance for {date} deleted.'})

@app.route('/dashboard')
def dashboard():
    return render_template('index.html')

@app.route('/dashboard_stats')
def dashboard_stats():
    today = datetime.now().strftime('%Y-%m-%d')
    records = query_attendance_records(date=today)
    present = sum(1 for r in records if r['status'].lower() == 'present')
    absent = sum(1 for r in records if r['status'].lower() == 'absent')
    late = sum(1 for r in records if r['status'].lower() == 'late')
    # Total = total registered students
    total = 0
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'r') as f:
            try:
                data = json.load(f)
                total = len(data)
            except Exception:
                total = 0
    return jsonify({
        'status': 'success',
        'present': present,
        'absent': absent,
        'late': late,
        'total': total
    })

@app.route('/recent_activity')
def recent_activity():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT name, status, date, time FROM attendance ORDER BY date DESC, time DESC LIMIT 10')
    rows = c.fetchall()
    conn.close()
    activity = [
        {'name': row[0], 'status': row[1], 'date': row[2], 'time': row[3]}
        for row in rows
    ]
    return jsonify({'status': 'success', 'activity': activity})

@app.route('/notifications')
def notifications():
    # For now, use recent attendance and schedule events as notifications
    notes = []
    # Attendance events
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT name, status, date, time FROM attendance ORDER BY date DESC, time DESC LIMIT 5')
    for row in c.fetchall():
        notes.append({
            'type': 'attendance',
            'message': f"{row[0]} marked {row[1]}",
            'date': row[2],
            'time': row[3]
        })
    conn.close()
    # Schedule events (from schedule_config.json)
    if os.path.exists(SCHEDULE_CONFIG_FILE):
        with open(SCHEDULE_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            if 'attendance_start_time' in config and 'attendance_end_time' in config:
                notes.append({
                    'type': 'schedule',
                    'message': f"Attendance scheduled from {config['attendance_start_time']} to {config['attendance_end_time']}",
                    'date': datetime.now().strftime('%Y-%m-%d'),
                    'time': config['attendance_start_time']
                })
    notes = sorted(notes, key=lambda n: (n['date'], n['time']), reverse=True)[:8]
    return jsonify({'status': 'success', 'notifications': notes})

@app.route('/current_user')
def current_user():
    # Demo user for now
    user = session.get('user', {'name': 'Admin User', 'role': 'Administrator', 'avatar': 'AD'})
    return jsonify({'status': 'success', 'user': user})

SETTINGS_FILE = 'settings.json'

def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        # Default: password is 'neards8book', email notifications off
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({'password': 'neards8book', 'email_notifications': False}, f)
    with open(SETTINGS_FILE, 'r') as f:
        return json.load(f)

def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

@app.route('/change_password', methods=['POST'])
def change_password():
    data = request.get_json()
    current = data.get('current')
    newpw = data.get('newpw')
    settings = load_settings()
    if current != settings.get('password'):
        return jsonify({'status': 'error', 'message': 'Current password is incorrect.'})
    if not newpw or len(newpw) < 6:
        return jsonify({'status': 'error', 'message': 'New password must be at least 6 characters.'})
    settings['password'] = newpw
    save_settings(settings)
    return jsonify({'status': 'success', 'message': 'Password changed successfully.'})

@app.route('/get_notification_settings')
def get_notification_settings():
    settings = load_settings()
    return jsonify({'status': 'success', 'email': settings.get('email_notifications', False)})

@app.route('/set_notification_settings', methods=['POST'])
def set_notification_settings():
    data = request.get_json()
    email = data.get('email', False)
    settings = load_settings()
    settings['email_notifications'] = bool(email)
    save_settings(settings)
    return jsonify({'status': 'success', 'message': 'Notification settings updated.'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    settings = load_settings()
    # Only allow teacher/actual password
    if username == 'teacher' and password == settings.get('password', 'neards8book'):
        session['user'] = {'name': username, 'role': 'Administrator', 'avatar': username[:2].upper()}
        return jsonify({'status': 'success', 'message': 'Logged in'})
    return jsonify({'status': 'error', 'message': 'Invalid credentials'})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return jsonify({'status': 'success', 'message': 'Logged out'})

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/delete_attendance_record', methods=['POST'])
def delete_attendance_record():
    data = request.get_json()
    date = data.get('date')
    name = data.get('name')
    if not date or not name:
        return jsonify({'status': 'error', 'message': 'Date and name required.'})
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('DELETE FROM attendance WHERE date = ? AND name = ?', (date, name))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': f'Record for {name} on {date} deleted.'})

if __name__ == '__main__':
    app.run(debug=True)