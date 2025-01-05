import pandas as pd
import random
import string
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer

# Unduh data NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

# Membaca file CSV
data = pd.read_csv('virtual_audio_assistant_questions.csv', encoding='utf-8')

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Membuat list pertanyaan dan jawaban dari CSV
audio_questions = data['Question'].tolist()
audio_answers = data['Answer'].tolist()

# Keyword
GREETING_INPUTS = ("halo", "hi", "hai", "hello", "hei")
GREETING_RESPONSES = [
    "Halo! Ada yang ingin kamu tanyakan tentang audio?",
    "Hi! Aku di sini untuk membantu dengan pertanyaan audio kamu.",
    "Halo, ada perangkat audio yang kamu ingin tahu lebih banyak?"
]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

def build_vsm_responses(user_response):
    """Menggunakan Vector Space Model untuk mencari kecocokan."""
    robo_response = ''

    # Menambahkan input pengguna ke daftar pertanyaan untuk diproses
    audio_questions.append(user_response)

    # Membuat representasi vektor TF-IDF
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf_matrix = TfidfVec.fit_transform(audio_questions)

    # Menghitung kesamaan kosinus
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    similarity_scores = cosine_similarities.flatten()
    sorted_indices = similarity_scores.argsort()[::-1][1:]

    # Mendapatkan hasil terbaik
    if similarity_scores[sorted_indices[0]] == 0:
        robo_response = "Maaf, aku belum punya jawaban untuk itu. Coba tanyakan dengan cara lain."
    else:
        best_match_index = sorted_indices[0]
        robo_response = audio_answers[best_match_index]

    # Menghapus input pengguna dari list pertanyaan untuk menjaga konsistensi data
    audio_questions.pop()
    return robo_response

ADDITIONAL_RESPONSES = {
    "siapa kamu": "Aku adalah asisten virtual Audiophilez. Tanyakan apa saja tentang headphone, speaker, atau pengaturan suara.",
    "terima kasih": "Sama-sama! Aku senang bisa membantu.",
    "selamat tinggal": "Selamat tinggal! Jika butuh saran audio lagi, aku di sini.",
}

def response(user_response):
    user_response = user_response.lower()
    if user_response in ADDITIONAL_RESPONSES:
        return ADDITIONAL_RESPONSES[user_response]
    if greeting(user_response):
        return greeting(user_response)
    # Memberikan efek "loading"
    print("...")
    time.sleep(3)  # Tambahkan jeda 3 detik untuk efek loading
    return build_vsm_responses(user_response)
