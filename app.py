from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json, os, threading, time
import mysql.connector
from werkzeug.security import check_password_hash
from werkzeug.utils import secure_filename
from one import rag_answer_final, refresh_faiss_index_if_updated

app = Flask(__name__)
app.secret_key = "change_this_to_a_strong_secret"

# ---------------- إعدادات ----------------
DATA_FILE = os.path.join(os.path.dirname(__file__), "kk.json")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ---------------- دوال إدارة البيانات ----------------
def load_data():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ---------------- تحديث RAG Index فورًا ----------------
def refresh_rag_index():
    """
    إعادة تحميل البيانات للـ RAG index
    """
    global rag_index
    rag_index = load_data()

# تحميل البيانات عند تشغيل السيرفر
rag_index = load_data()

# ---------------- دوال قاعدة البيانات ----------------
def get_db_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="rag_system"
    )
    return conn

def add_student_db(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return False
    cursor.execute(
        "INSERT INTO users (username, password, user_type, user_state) VALUES (%s, %s, %s, %s)",
        (username, password, 'student', 'active')
    )
    conn.commit()
    cursor.close()
    conn.close()
    return True

def check_user_db(username, password):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username=%s", (username,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if user:
        if user['user_type'].strip().lower() == 'admin':
            if check_password_hash(user['password'], password):
                return user
        else:
            if user['password'] == password:
                return user
    return None

# ---------------- الصفحات ----------------
@app.route('/')
def home():
    return redirect(url_for('loginstu'))

@app.route("/loginstu", methods=["GET", "POST"], endpoint="loginstu")
def student_login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = check_user_db(username, password)
        if user:
            session["username"] = username
            session["user_type"] = user['user_type'].strip().lower()
            if session["user_type"] == 'admin':
                return redirect(url_for("admin_page"))
            else:
                return redirect(url_for("index"))
        else:
            return render_template("loginstu.html", error="❌ اسم المستخدم أو كلمة المرور غير صحيحة")
    return render_template("loginstu.html")

@app.route("/create_account", methods=["GET", "POST"])
def create_account():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        confirm = request.form.get("confirm_password", "").strip()
        if password != confirm:
            return render_template("createaccount.html", error="⚠️ كلمتا المرور غير متطابقتين")
        if add_student_db(username, password):
            session["username"] = username
            session["user_type"] = "student"
            return redirect(url_for("index"))
        else:
            return render_template("createaccount.html", error="⚠️ اسم المستخدم موجود بالفعل")
    return render_template("createaccount.html")

@app.route("/index")
def index():
    if "username" not in session or session.get("user_type") != "student":
        return redirect(url_for("loginstu"))
    return render_template("index.html", username=session["username"])

@app.route("/logout")
def logout():
    session.pop("username", None)
    session.pop("user_type", None)
    return redirect(url_for("loginstu"))

@app.route('/index_public')
def index_public():
    return render_template("index.html", username="زائر")

# ---------------- نظام RAG ----------------
@app.route('/ask', methods=['POST'])
def ask():
    payload = request.get_json()
    question = payload.get("question", "")
    # استخدام RAG الذكي مع FAISS
    answer_text = rag_answer_final(question)
    return jsonify({"answer": answer_text})

# ---------------- واجهة الأدمن ----------------
@app.route('/admin')
def admin_page():
    if "username" not in session or session.get("user_type") != "admin":
        return redirect(url_for("loginstu"))
    data = load_data()
    return render_template('admin.html', data=data)

# ---------------- CRUD ----------------
@app.route('/add', methods=['POST'])
def add_item():
    if session.get("user_type") != "admin":
        return jsonify(success=False), 401

    content = request.form.get('content', '').strip()
    file = request.files.get('file')
    file_url = None

    if file:
        filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_url = url_for('static', filename=f"uploads/{filename}")

    data = load_data()
    next_id = max([item.get("id", 0) for item in data], default=0) + 1

    new_item = {"id": next_id, "content": content, "file_url": file_url}
    data.append(new_item)
    save_data(data)

    refresh_rag_index()  # تحديث الـ RAG فورًا

    return jsonify(success=True, new_item=new_item)

@app.route('/update', methods=['POST'])
def update_item():
    if session.get("user_type") != "admin":
        return jsonify(success=False), 401

    data = load_data()
    try:
        item_id = int(request.form.get('id', -1))
        new_content = request.form.get('content', '').strip()
    except ValueError:
        return jsonify(success=False, message="Invalid ID")

    for item in data:
        if item.get("id") == item_id:
            item["content"] = new_content
            save_data(data)
            refresh_rag_index()  # تحديث الـ RAG فورًا
            return jsonify(success=True, data=data)

    return jsonify(success=False, message="Item not found")

@app.route('/delete/<int:item_id>', methods=['POST'])
def delete_item(item_id):
    if session.get("user_type") != "admin":
        return jsonify(success=False), 401

    data = load_data()
    data = [item for item in data if item.get("id") != item_id]

    # إعادة ترقيم الـ IDs
    for index, item in enumerate(data, start=1):
        item["id"] = index

    save_data(data)
    refresh_rag_index()  # تحديث الـ RAG فورًا
    return jsonify(success=True)

@app.route('/search')
def search():
    q = request.args.get('q', '').strip().lower()
    by = request.args.get('by', 'text')
    data = load_data()
    if not q:
        return jsonify(data)
    if by == 'id':
        try:
            q_id = int(q)
            results = [item for item in data if item.get("id") == q_id]
        except ValueError:
            results = []
    else:
        results = [item for item in data if q in item.get("content", "").lower()]
    return jsonify(results)

# ---------------- تشغيل ----------------
if __name__ == "__main__":
    app.run(debug=True)
