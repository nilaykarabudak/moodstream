# 🎬 MoodStream: Yayın Analiz Asistanı  
**Bitirme Projesi** | İstinye Üniversitesi | Haziran 2025  

##  Özet  
Bu proje, Türkçe dijital yayın platformlarına gelen kullanıcı yorumlarını;  
**Kategori**, **Alt Kategori**, **Segment** ve **Duygu Analizi** başlıkları altında otomatik olarak sınıflandıran bir yapay zeka uygulamasıdır.  
Streamlit tabanlı kullanıcı arayüzü ile gerçek zamanlı ve kullanıcı dostu analiz deneyimi sunmaktadır.

##  Kullanılan Teknolojiler  
- **Programlama Dili:** Python  
- **Makine Öğrenmesi:** Scikit-learn, Joblib  
- **Doğal Dil İşleme:** NLTK, TF-IDF  
- **Model:** Logistic Regression  
- **Veri Görselleştirme:** Matplotlib  
- **Arayüz:** Streamlit  


##  Klasör Yapısı  
```plaintext
/btr_projesi  
├── Yorum 5 1 (1).xlsx              # Ana veri seti  
├── app.py                          # Streamlit arayüz dosyası  
├── model_egit.py                   # Model eğitimi scripti  
├── kategori_model.pkl              # Ana kategori modeli  
├── altkategori_model.pkl           # Alt kategori modeli  
├── segment_model.pkl               # Segment modeli  
├── sentiment_model.pkl             # Duygu analizi modeli  
├── kategori_vectorizer.pkl         # TF-IDF vektörizer (kategori)  
├── altkategori_vectorizer.pkl      # TF-IDF vektörizer (alt kategori)  
├── segment_vectorizer.pkl          # TF-IDF vektörizer (segment)  
├── sentiment_vectorizer.pkl        # TF-IDF vektörizer (duygu)  
├── README.md                       # Proje açıklaması  
└── bitirme_projesi_model_dosyalari.zip   # Arşivlenmiş modeller
```

## 👩 Geliştirici  
**Nilay Zehra Karabudak**  
İstinye Üniversitesi - Bilgisayar Programcılığı  
2025 Mezuniyet Projesi
