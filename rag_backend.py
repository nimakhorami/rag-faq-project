import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import joblib
import re

class SimpleRetriever:
    def __init__(self, cache_dir="cache"):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_paths(self, file_path):
        base = os.path.splitext(os.path.basename(file_path))[0]
        vec_path = os.path.join(self.cache_dir, f"{base}_vectorizer.joblib")
        mat_path = os.path.join(self.cache_dir, f"{base}_matrix.joblib")
        return vec_path, mat_path

    def load_documents(self, file_path):
        vec_path, mat_path = self._get_cache_paths(file_path)

        # بارگذاری از کش اگر موجود باشه
        if os.path.exists(vec_path) and os.path.exists(mat_path):
            print("⚡ بارگذاری مدل از کش...")
            self.vectorizer = joblib.load(vec_path)
            self.tfidf_matrix = joblib.load(mat_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
            print(f"📚 {len(self.documents)} قطعه متنی از فایل اصلی بارگذاری شد")
            print("✅ مدل از کش بارگذاری شد")
            return

        # در غیر این صورت از فایل اصلی بخون و مدل بساز
        print("🔨 ساخت مدل جدید...")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        self.documents = [doc.strip() for doc in content.split('\n\n') if doc.strip()]
        print(f"📚 {len(self.documents)} قطعه متنی بارگذاری شد")

        if self.documents:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
            joblib.dump(self.vectorizer, vec_path)
            joblib.dump(self.tfidf_matrix, mat_path)
            print("💾 مدل در کش ذخیره شد")
        else:
            print("⚠️ هیچ سندی بارگذاری نشد!")

    def search(self, query, top_k=3):
        if not self.documents or self.tfidf_matrix is None:
            return "سیستم آماده نیست. لطفاً فایل دانش را بررسی کنید."

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:
                results.append({
                    'text': self.documents[idx],
                    'score': float(similarities[idx])
                })

        if results:
            best_text = results[0]['text']

            # استخراج فقط بخش پاسخ از قالب FAQ
            match = re.search(r'پاسخ:\s*(.*)', best_text, re.DOTALL)
            if match:
                best_text = match.group(1).strip()

            return best_text
        else:
            return "متأسفانه پاسخی برای سوال شما پیدا نشد."

retriever = SimpleRetriever()

def initialize_rag(file_path="faq.txt"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, file_path)

    retriever.load_documents(full_path)
    return retriever

def search_knowledge(question):
    return retriever.search(question)
