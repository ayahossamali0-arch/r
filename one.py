import os
import json
import re
import time
import threading
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Tuple

# ==============================
# 0️⃣ تحميل الإعدادات الأساسية
# ==============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise ValueError("❌ لم يتم العثور على OPENAI_API_KEY في ملف .env")

client = OpenAI(api_key=OPENAI_API_KEY)

TOP_K = 5
SIMILARITY_THRESHOLD = 0.55
EMBED_MODEL_NAME = "intfloat/multilingual-e5-large"
DATA_FILE = os.path.join(os.path.dirname(__file__), "kk.json")

# ==============================
# 1️⃣ تحميل البيانات وبناء FAISS
# ==============================
def load_data():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print("⚠️ kk.json تالف — إعادة التهيئة...")
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

data = load_data()
texts = [item["content"] for item in data] if data else ["لا توجد بيانات بعد."]
model = SentenceTransformer(EMBED_MODEL_NAME)
text_embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

dimension = text_embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efSearch = 64
index.add(text_embeddings)
last_modified = os.path.getmtime(DATA_FILE)

# ==============================
# 2️⃣ تحديث تلقائي للـ FAISS
# ==============================
index_lock = threading.Lock()

def refresh_faiss_index_if_updated():
    global last_modified, index, text_embeddings, texts
    current_modified = os.path.getmtime(DATA_FILE)
    if current_modified != last_modified:
        print("🔄 اكتشاف تعديل في kk.json — تحديث الفهرس...")
        new_data = load_data()
        if not new_data:
            print("⚠️ لا توجد بيانات جديدة بعد.")
            return
        new_texts = [item["content"] for item in new_data]
        new_embeddings = model.encode(new_texts, normalize_embeddings=True).astype("float32")
        new_index = faiss.IndexHNSWFlat(new_embeddings.shape[1], 32)
        new_index.hnsw.efSearch = 64
        new_index.add(new_embeddings)
        with index_lock:
            index = new_index
            text_embeddings = new_embeddings
            texts = new_texts
            last_modified = current_modified
        print(f"✅ تم تحديث الفهرس ({len(texts)} عناصر).")

# ==============================
# 3️⃣ تجهيز النصوص والنوايا
# ==============================
ARABIC_DIACRITICS = re.compile(r"[ًٌٍَُِّْـ]")

def normalize_arabic(text: str) -> str:
    text = re.sub(ARABIC_DIACRITICS, "", text)
    text = text.replace("آ", "ا").replace("أ", "ا").replace("إ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================
# 🔥 ذكاء فهم النوايا
# ==============================
def detect_special_intent(text: str) -> Tuple[bool, str]:
    text_low = text.lower().strip()

    greeting_patterns = [
        r"مرحب", r"اهلا", r"السلام", r"هلا", r"صباح الخير", r"مساء الخير",
        r"hi", r"hello", r"hey"
    ]
    for p in greeting_patterns:
        if re.search(p, text_low): return True, "greeting"

    farewell_patterns = [
        r"مع السلامة", r"سلام", r"باي", r"وداع", r"الى اللقاء", r"goodbye", r"bye"
    ]
    for p in farewell_patterns:
        if re.search(p, text_low): return True, "farewell"

    thanks_patterns = [
        r"شكر", r"ممنون", r"thx", r"thank", r"يسلمو"
    ]
    for p in thanks_patterns:
        if re.search(p, text_low): return True, "thanks"

    praise_patterns = [
        r"رائع", r"ممتاز", r"جميل", r"ذكي", r"عبقري", r"احسنت"
    ]
    for p in praise_patterns:
        if re.search(p, text_low): return True, "praise"

    hate_patterns = [
        r"اكرهك", r"بكرهك", r"hate you", r"i hate you"
    ]
    for p in hate_patterns:
        if re.search(p, text_low): return True, "hate"

    return False, ""


# ==============================
# 4️⃣ سؤال الطالب والرد
# ==============================
def rag_answer_final(user_question: str) -> str:
    threading.Thread(target=refresh_faiss_index_if_updated, daemon=True).start()

    # ❗ كشف النصوص غير المفهومة هنا
    n = user_question.strip()

    if len(n) < 3:
        return "❗ النص غير واضح، يرجى توضيح سؤالك."

    if re.fullmatch(r"[^A-Za-z0-9\u0600-\u06FF]+", n):
        return "❗ النص غير واضح، يرجى توضيح سؤالك."

    if re.fullmatch(r"[a-zA-Z]{3,}", n) and not re.search(r"(what|when|where|why|how)", n):
        return "❗ النص غير واضح، يرجى توضيح سؤالك."

    if re.fullmatch(r"[أ-ي]{3,}", n) and len(set(n)) <= 2:
        return "❗ النص غير واضح، يرجى توضيح سؤالك."

    # 👇 ردود النوايا
    intent_found, intent_type = detect_special_intent(user_question)
    if intent_found:
        responses = {
            "greeting": "👋 أهلاً بك! كيف أقدر أساعدك اليوم؟",
            "farewell": "👋 مع السلامة! أتمنى لك يوم جميل.",
            "thanks": "🤗 لا شكر على واجب! بالتوفيق دائمًا.",
            "praise": "🙏 شكراً على ذوقك! أنا هنا علشان أساعدك.",
            "hate": "🙂 خلينا نبدأ من جديد… أنا هنا لمساعدتك فقط ❤️",
        }
        return responses.get(intent_type, "🙂 حاضر.")

    # تحسين فهم السؤال باستخدام GPT
    try:
        ai_understanding = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "أعد صياغة السؤال بشكل أوضح لغرض البحث داخل قاعدة بيانات نصية."},
                {"role": "user", "content": user_question},
            ]
        )
        refined_question = ai_understanding.choices[0].message.content.strip()
    except Exception:
        refined_question = user_question

    search_query = normalize_arabic(refined_question + " " + user_question)

    # مطابقة مباشرة مع قاعدة البيانات
    for item in load_data():
        content_norm = normalize_arabic(item.get("content", ""))
        file_url = item.get("file_url", "")
        if search_query in content_norm or (file_url and search_query in file_url.lower()):
            if file_url:
                return f"<img src='{file_url}' style='max-width:300px;'>"
            return item.get("content", "")

    # بحث المتجهات
    try:
        q_emb = model.encode([search_query], normalize_embeddings=True).astype("float32")
        with index_lock:
            distances, indices = index.search(q_emb, TOP_K)

        results = [(int(idx), float(1 - dist)) for idx, dist in zip(indices[0], distances[0])]
        results = [r for r in results if r[0] >= 0]
        results = sorted(results, key=lambda x: x[1], reverse=True)

        if not results or results[0][1] < SIMILARITY_THRESHOLD:
            return "❗ لا توجد أي معلومة عن هذا السؤال في قاعدة البيانات الحالية."

        best_idx = results[0][0]
        data_list = load_data()
        best_item = data_list[best_idx]

        if best_item.get("file_url"):
            return f"<img src='{best_item['file_url']}' style='max-width:300px;'>"

        try:
            optimized_answer = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "قدّم جواباً واضحاً ومباشراً بناءً على النص المعطى."},
                    {"role": "user", "content": f"السؤال: {user_question}\n\nالنص:\n{best_item.get('content', '')}"},
                ]
            )
            return optimized_answer.choices[0].message.content.strip()
        except Exception:
            return best_item.get("content", "")

    except Exception as e:
        return f"⚠️ حدث خطأ أثناء الإجابة: {e}"


# ==============================
# 5️⃣ تشغيل تفاعلي
# ==============================
if __name__ == "__main__":
    print("🤖 نظام RAG الذكي جاهز! اكتب سؤالك بالعربية.")
    print("🟢 اكتب 'خروج' لإنهاء الجلسة.")
    while True:
        user_q = input("🧑‍🎓: ").strip()
        if user_q.lower() in ["خروج", "exit", "quit"]:
            print("🤖: مع السلامة 👋")
            break
        print("🤖:", rag_answer_final(user_q))
