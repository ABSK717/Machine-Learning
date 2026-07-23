import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import pickle

# Sample dataset
data = {
    "text": [
        "I absolutely loved the movie! It was fantastic.",
        "The service at the restaurant was excellent.",
        "I am so happy with my new phone—it works perfectly!",
        "What a wonderful experience! I would definitely go again.",
        "The customer support was really helpful and kind.",
        "This product exceeded my expectations!",
        "Such a delightful book, I couldn’t put it down.",
        "I highly recommend this place! The ambiance is amazing.",
        "This app is so user-friendly and efficient!",
        "My order arrived on time, and everything was perfect.",
        "The food was delicious, and the presentation was beautiful.",
        "I had a great day at the amusement park!",
        "This is one of the best laptops I’ve ever used.",
        "The vacation was relaxing and absolutely stress-free.",
        "I’m so grateful for the support from my team!",
        "The concert was electrifying! I had so much fun.",
        "The coffee here is amazing, and the staff is super friendly.",
        "This gaming experience is so immersive and exciting.",
        "I feel so productive after using this software.",
        "What a fantastic piece of technology!",
        
        "The movie was boring, and the plot made no sense.",
        "The restaurant service was extremely slow and disappointing.",
        "I regret buying this phone; it keeps freezing.",
        "What a terrible experience! I will never go back there.",
        "The product was nothing like the description.",
        "I’m very unhappy with my purchase.",
        "The book was so dull, I couldn’t finish it.",
        "This app crashes all the time. So frustrating!",
        "My package arrived late and was damaged.",
        "The food was cold and had no flavor.",
        "The theme park was overcrowded and overpriced.",
        "This laptop is slow and unreliable.",
        "The hotel room was dirty and smelled bad.",
        "I’m extremely disappointed with this service.",
        "The concert was a mess—bad sound and poor organization.",
        "This coffee tastes awful and is overpriced.",
        "The controls in this game are horrible.",
        "I feel so unproductive using this buggy software.",
        "The customer support was rude and unhelpful.",
        "This technology is outdated and useless.",
        
        "The movie was okay, nothing special.",
        "The restaurant had an average menu, nothing extraordinary.",
        "The phone works fine, but it’s nothing groundbreaking.",
        "My experience at the store was neither good nor bad.",
        "The product is decent but has room for improvement.",
        "The book had some good parts but was overall just okay.",
        "This app is functional but lacks key features.",
        "My order was slightly delayed but arrived in good condition.",
        "The food was edible, but I wouldn’t order it again.",
        "The service was standard—nothing to complain about, but not great either."
    ],
    "sentiment": [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]  # 1: Positive, 0: Negative, 2: Neutral
}


df = pd.DataFrame(data)

# Text preprocessing
vectorizer = TfidfVectorizer(max_features=500)
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["sentiment"]

# Convert to Training Data
X = data["text"]
y = data["sentiment"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build SVM Model with TF-IDF
model = make_pipeline(TfidfVectorizer(), SVC(kernel="linear"))

# Train Model
model.fit(X_train, y_train)


# Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
#model = MultinomialNB()
#model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model & vectorizer
with open("sentiment_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("done")