# add_admin.py
import mysql.connector
from werkzeug.security import generate_password_hash

# -----------------------------
# إعداد بيانات الأدمن
# -----------------------------
admin_username = "admin"         # ← اكتب اسم الأدمن هنا
admin_password = "1234"          # ← اكتب كلمة المرور التي تريدها هنا
admin_state = "active"           # ← الحالة (active أو inactive)

# -----------------------------
# الاتصال بقاعدة البيانات
# -----------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # ضع كلمة مرور MySQL إذا كانت موجودة
    database="rag_system"  # ← اسم قاعدة البيانات
)
cursor = conn.cursor()

# -----------------------------
# تجزئة كلمة المرور (للأمان)
# -----------------------------
hashed_password = generate_password_hash(admin_password)

# -----------------------------
# إضافة الأدمن أو تحديثه إن كان موجودًا
# -----------------------------
cursor.execute("""
    INSERT INTO users (username, password, user_type, user_state)
    VALUES (%s, %s, 'admin', %s)
    ON DUPLICATE KEY UPDATE 
        password = VALUES(password),
        user_state = VALUES(user_state),
        user_type = 'admin'
""", (admin_username, hashed_password, admin_state))

conn.commit()
conn.close()

print(f"✅ تم إضافة أو تحديث الأدمن بنجاح:")
print(f"اسم المستخدم: {admin_username}")
print(f"كلمة المرور: {admin_password} (تُحفظ مجزأة في قاعدة البيانات)")
