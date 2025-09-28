import pandas as pd
import string
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Elle tanımlanmış Türkçe stopwords listesi
stop_words = set("""
acaba ama aslında az bazı bazen bile bütün çünkü çok da daha de defa değil diye gibi hep hiçbir için ile ise kez ki kim mi mu ne neden nasıl o sanki se şey siz sonra şu tüm ve veya ya yani
""".split())

def temizle(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    kelimeler = text.split()
    kelimeler = [k for k in kelimeler if k not in stop_words]
    return " ".join(kelimeler)

# Excel verisini oku
excel_dosya = "Yorum 5 1 (1).xlsx"
df = pd.read_excel(excel_dosya)

# Hatalı etiket düzeltmeleri
duzeltmeler = {
    'Yüsek Fiyat': 'Yüksek Fiyat',
    'YüksekFiyat': 'Yüksek Fiyat',
    'Abnelik': 'Abonelik',
    'Kullanım Kolayllığı': 'Kullanım Kolaylığı'
}

etiketler = ["Kategori", "AltKategori", "Segment", "Sentiment"]
for etiket in etiketler:
    if etiket in df.columns:
        df[etiket] = df[etiket].replace(duzeltmeler)

# Model eğitimi fonksiyonu
def model_egit(sutun_adi, model_ad):
    sub_df = df.dropna(subset=["Yorum", sutun_adi])
    frekanslar = sub_df[sutun_adi].value_counts()
    gecerli_siniflar = frekanslar[frekanslar >= 5].index
    sub_df = sub_df[sub_df[sutun_adi].isin(gecerli_siniflar)]

    if len(sub_df) < 30:
        print(f"{sutun_adi} için yeterli veri yok.")
        return

    sub_df["temiz_yorum"] = sub_df["Yorum"].apply(temizle)

    tfidf = TfidfVectorizer(max_df=0.95, min_df=2, ngram_range=(1,2))
    X = tfidf.fit_transform(sub_df["temiz_yorum"])
    y = sub_df[sutun_adi]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"✔ {sutun_adi} Doğruluk: {acc:.2f}")
    print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred, zero_division=0))

    joblib.dump(model, model_ad)
    joblib.dump(tfidf, f"{sutun_adi.lower()}_vectorizer.pkl")

# Eğitim başlat
model_egit("Kategori", "kategori_model.pkl")
model_egit("AltKategori", "altkategori_model.pkl")
model_egit("Segment", "segment_model.pkl")
model_egit("Sentiment", "sentiment_model.pkl")

print("\n✅ Tüm modeller başarıyla eğitildi.")

