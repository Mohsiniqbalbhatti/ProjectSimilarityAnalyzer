from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5EncoderModel
from sklearn.metrics.pairwise import cosine_similarity
import redis
import logging
from werkzeug.security import check_password_hash, generate_password_hash
import json
import jwt as pyjwt
import datetime
from functools import wraps

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.secret_key = 'your-secret-key-123'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///projects.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Enable CORS for http://localhost:5173
CORS(app, supports_credentials=True, resources={r"*": {"origins": "http://localhost:5173"}})

# Database setup
db = SQLAlchemy(app)

# Redis setup
try:
    cache = redis.Redis(host='localhost', port=6379, db=0)
    cache.ping()
    logger.info("Successfully connected to Redis.")
except redis.ConnectionError as e:
    logger.warning(f"Failed to connect to Redis: {e}. Caching will be disabled.")
    cache = None

# Load AI models
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small')
t5_model = T5EncoderModel.from_pretrained('t5-small')

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    domain = db.Column(db.String(50), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    summary = db.Column(db.Text, nullable=False)
    domain = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    results = db.Column(db.JSON, nullable=True)  # New field for storing similarity results
    user = db.relationship('User', backref=db.backref('histories', lazy=True))

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    history_id = db.Column(db.Integer, db.ForeignKey('history.id'), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref=db.backref('feedbacks', lazy=True))
    history = db.relationship('History', backref=db.backref('feedbacks', lazy=True))

class Contact(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __repr__(self):
        return f'<Contact {self.name}>'


# Helper: Token Required Decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.cookies.get('token')
        if not token:
            return jsonify({'error': 'Token is missing!'}), 401
        try:
            data = pyjwt.decode(token, app.secret_key, algorithms=["HS256"])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                raise Exception('User not found')
        except pyjwt.ExpiredSignatureError:
            return jsonify({'error': 'Token has expired'}), 401
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Preprocess and similarity logic (same as before)
def preprocess_text(text):
    return text.lower().strip()

def generate_t5_features(texts):
    inputs = t5_tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    outputs = t5_model(**inputs).last_hidden_state.mean(dim=1)
    return outputs.detach().numpy()

def calculate_similarity(title, summary):
    cache_key = f"similarity_{title}_{summary}"
    if cache and cache.ping():
        cached_result = cache.get(cache_key)
        if cached_result:
            cached_data = json.loads(cached_result.decode('utf-8'))
            top_two = []
            for item in cached_data:
                proj = Project.query.get(item['id'])
                if proj:
                    top_two.append((proj, item['similarity']))
            return top_two

    user_input = preprocess_text(f"{title} {summary} {summary}")
    user_sbert_features = sbert_model.encode([user_input])
    user_t5_features = generate_t5_features([user_input])
    user_features = np.concatenate([user_sbert_features, user_t5_features], axis=1)

    projects = Project.query.all()
    scores = []
    for proj in projects:
        proj_text = preprocess_text(f"{proj.title} {proj.summary} {proj.summary}")
        proj_sbert_features = sbert_model.encode([proj_text])
        proj_t5_features = generate_t5_features([proj_text])
        proj_features = np.concatenate([proj_sbert_features, proj_t5_features], axis=1)
        sim_score = cosine_similarity(user_features, proj_features)[0][0]
        similarity_percentage = (sim_score + 1) / 2 * 100
        scores.append((proj, similarity_percentage))

    scores.sort(key=lambda x: x[1], reverse=True)
    top_two = scores[:2]

    if cache and cache.ping():
        cache_data = [{'id': proj.id, 'similarity': float(similarity)} for proj, similarity in top_two]
        cache.set(cache_key, json.dumps(cache_data), ex=3600)
    return top_two

# Routes
@app.route('/')
def home():
    return jsonify({"message": "Welcome to the API. Use /login to authenticate."})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    identifier = data.get('identifier', '').strip()
    password = data.get('password', '').strip()

    user = User.query.filter((User.email == identifier) | (User.username == identifier)).first()

    if user and user.check_password(password):
        token = pyjwt.encode({
            'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        }, app.secret_key, algorithm='HS256')

        resp = make_response(jsonify({"message": "Login successful"}))

        # 🔥 Fix: Make sure cookie is actually set
        resp.set_cookie(
            'token',
            token,
            httponly=True,
            secure=False,  # Set True for HTTPS
            samesite='Lax'  # Change from 'Strict' to 'Lax'
        )

        return resp

    return jsonify({"error": "Invalid credentials"}), 401

@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()

    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    # Check for missing fields
    if not name or not email or not username or not password:
        return jsonify({"error": "All fields  are required."}), 400

    # Check for duplicate email or username
    if User.query.filter_by(email=email).first():
        return jsonify({"error": "Email is already registered."}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username is already taken."}), 400

    # Create and save user
    new_user = User(name=name, email=email, username=username)
    new_user.set_password(password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Signup successful"}), 200


@app.route('/logout')
def logout():
    resp = make_response(jsonify({"message": "Logged out"}))
    resp.set_cookie('token', '', expires=0)
    return resp

@app.route('/user', methods=['GET'])
@token_required
def get_user(current_user):
    return jsonify({
        "status": "success",
        "user": {
            "name": current_user.name,
            "email": current_user.email,
            "username": current_user.username
        }
    }), 200


@app.route('/user/updateEmail', methods=['POST'])
@token_required
def update_email(current_user):
    data = request.get_json()
    current_password = data.get('password', '').strip()
    new_email = data.get('newEmail', '').strip()

    # Check if the current password is correct
    if not current_user.check_password(current_password):
        return jsonify({"error": "Current password is incorrect"}), 400

    # Validate the new email
    if not new_email:
        return jsonify({"error": "New email is required"}), 400

    # Check if the new email is already in use
    existing_user = User.query.filter_by(email=new_email).first()
    if existing_user:
        return jsonify({"error": "Email is already in use"}), 400

    # Update the user's email
    current_user.email = new_email
    db.session.commit()

    return jsonify({"message": "Email updated successfully"}), 200

@app.route('/user/updatePassword', methods=['POST'])
@token_required
def update_password(current_user):
    app.logger.info("Update password route hit")
    data = request.get_json()
    current_password = data.get('password', '').strip()
    new_password = data.get('newPassword', '').strip()

    # Check if the current password is correct
    if not current_user.check_password(current_password):
        return jsonify({"error": "Current password is incorrect"}), 400

    # Validate the new password
    if not new_password or len(new_password) < 6:
        return jsonify({"error": "New password must be at least 6 characters long"}), 400

    # Update the user's password
    current_user.set_password(new_password)
    db.session.commit()

    return jsonify({"message": "Password updated successfully"}), 200

@app.route('/index', methods=['POST'])
@token_required
def index(current_user):
    data = request.get_json()
    title = data.get('title', '').strip()
    summary = data.get('summary', '').strip()
    domain = data.get('domain', '').strip()
    history_id = data.get('historyId', None)  # Get the historyId from the request

    # Calculate similarity between the new project and existing projects
    top_matches = calculate_similarity(title, summary)

    if history_id:
        # If historyId is provided, update the existing history
        existing_history = History.query.filter_by(id=history_id, user_id=current_user.id).first()
        if existing_history:
            # Update the existing history entry with new title, summary, and domain
            existing_history.title = title
            existing_history.summary = summary
            existing_history.domain = domain
            # Update the results with the new similarity check
            existing_history.results = [{
                'id': proj.id,
                'title': proj.title,
                'summary': proj.summary,
                'domain': proj.domain,
                'similarity': similarity
            } for proj, similarity in top_matches]

            db.session.commit()
            return jsonify({
                "historyId": existing_history.id,  # Return the historyId of the updated history
                "matches": [{
                    'id': proj.id,
                    'title': proj.title,
                    'summary': proj.summary,
                    'domain': proj.domain,
                    'similarity': similarity
                } for proj, similarity in top_matches],
                "message": "Similarity Checked"
            })
        else:
            return jsonify({"error": "History not found"}), 404  # If no history entry is found
    else:
        # If no historyId is provided, create a new history entry
        new_history = History(
            user_id=current_user.id,
            title=title,
            summary=summary,
            domain=domain,
            results=[{
                'id': proj.id,
                'title': proj.title,
                'summary': proj.summary,
                'domain': proj.domain,
                'similarity': similarity
            } for proj, similarity in top_matches]
        )
        db.session.add(new_history)
        db.session.commit()

        return jsonify({
            "historyId": new_history.id,  # Return the historyId of the newly created history
            "matches": [{
                'id': proj.id,
                'title': proj.title,
                'summary': proj.summary,
                'domain': proj.domain,
                'similarity': similarity
            } for proj, similarity in top_matches],
            "message": "Similarity Checked"
        })

@app.route('/contact', methods=['POST'])
def contact():
    data = request.get_json()
    name = data.get('name')
    message = data.get('message')

    # Validate input
    if not name or not message:
        return jsonify({"error": "Name and message are required."}), 400

    # Create a new contact entry
    new_contact = Contact(
        name=name,
        message=message,
    )

    try:
        db.session.add(new_contact)
        db.session.commit()  # Save to database
        return jsonify({"message": "Messaged Sent!"}), 200
    except Exception as e:
        db.session.rollback()  # Rollback in case of error
        return jsonify({"error": str(e)}), 500  # Internal server error response


@app.route('/history', methods=['GET'])
@token_required
def history(current_user):
    user_history = History.query.filter_by(user_id=current_user.id).order_by(History.timestamp.desc()).all()
    history_data = [{
        "id": h.id,
        "title": h.title,
        "summary": h.summary,
        "domain": h.domain,
        "timestamp": h.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
        "results": h.results  # Include the similarity results here
    } for h in user_history]
    return jsonify({"history": history_data}), 200

@app.route('/history/<int:history_id>', methods=['GET', 'DELETE', 'PUT'])
@token_required
def manage_specific_history(current_user, history_id):
    history_entry = History.query.filter_by(id=history_id, user_id=current_user.id).first()
    if not history_entry:
        return jsonify({"error": "History not found"}), 404

    if request.method == 'GET':
        return jsonify({
            "id": history_entry.id,
            "title": history_entry.title,
            "summary": history_entry.summary,
            "domain": history_entry.domain,
            "results": history_entry.results,
            "timestamp": history_entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        }), 200
    elif request.method == 'DELETE':
        db.session.delete(history_entry)
        db.session.commit()
        return jsonify({"message": "Deleted"})
    elif request.method == 'PUT':
        data = request.get_json()
        history_entry.title = data.get('title', history_entry.title)
        history_entry.summary = data.get('summary', history_entry.summary)
        history_entry.domain = data.get('domain', history_entry.domain)
        db.session.commit()
        return jsonify({"message": "Updated successfully"})

@app.route('/api/similarity', methods=['POST'])
@token_required
def api_similarity(current_user):
    data = request.get_json()
    title = data.get('title', '').strip()
    summary = data.get('summary', '').strip()
    domain = data.get('domain', '').strip()
    top_matches = calculate_similarity(title, summary)
    results = [{
        'id': proj.id,
        'title': proj.title,
        'summary': proj.summary,
        'domain': proj.domain,
        'similarity': similarity
    } for proj, similarity in top_matches]
    return jsonify({"matches": results})

@app.route('/feedback', methods=['POST'])
@token_required
def feedback(current_user):
    data = request.get_json()
    feedback_message = data.get('message', '').strip()  # Feedback message
    history_id = data.get('historyId', None)  # History ID to associate feedback with

    # Check if both feedback message and historyId are provided
    if not feedback_message or not history_id:
        return jsonify({"error": "Both message and historyId are required"}), 400

    # Find the corresponding history entry using historyId and userId
    history = History.query.filter_by(id=history_id, user_id=current_user.id).first()

    if not history:
        return jsonify({"error": "History not found"}), 404  # If history entry is not found

    # Save the feedback message into the database (assuming a Feedback model exists)
    new_feedback = Feedback(
        user_id=current_user.id,
        history_id=history.id,  # Associate feedback with the project
        message=feedback_message,
    )
    db.session.add(new_feedback)
    db.session.commit()

    # Return a success response
    return jsonify({
        "message": "Thankyou for your Feedback!"
    }), 200


@app.route('/help')
def help():
    return jsonify({"message": "Available: /login, /logout, /signup, /index, /history, /user"})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='localhost', port=5000, debug=True)
