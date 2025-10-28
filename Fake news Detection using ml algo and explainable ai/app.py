{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f05bbea2-4af6-416c-9231-0273fcb0f4f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package punkt to\n",
      "[nltk_data]     C:\\Users\\User\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package punkt is already up-to-date!\n",
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\User\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package stopwords is already up-to-date!\n",
      "[nltk_data] Downloading package wordnet to\n",
      "[nltk_data]     C:\\Users\\User\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Package wordnet is already up-to-date!\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import joblib\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "from nltk.tokenize import word_tokenize\n",
    "from nltk.stem import WordNetLemmatizer\n",
    "import string\n",
    "\n",
    "# Download NLTK data\n",
    "nltk.download(\"punkt\")\n",
    "nltk.download(\"stopwords\")\n",
    "nltk.download(\"wordnet\")\n",
    "\n",
    "# Load model and vectorizer\n",
    "model = joblib.load(\"fake_news_xgb_model.pkl\")\n",
    "vectorizer = joblib.load(\"tfidf_vectorizer.pkl\")\n",
    "\n",
    "# Text cleaning function\n",
    "stop_words = set(stopwords.words(\"english\"))\n",
    "lemmatizer = WordNetLemmatizer()\n",
    "\n",
    "def clean_text(text):\n",
    "    tokens = word_tokenize(text.lower())\n",
    "    tokens = [t for t in tokens if t.isalpha()]\n",
    "    tokens = [t for t in tokens if t not in stop_words]\n",
    "    tokens = [lemmatizer.lemmatize(t) for t in tokens]\n",
    "    return \" \".join(tokens)\n",
    "\n",
    "# Streamlit app\n",
    "st.title(\"📰 Fake News Detector\")\n",
    "\n",
    "user_input = st.text_area(\"Enter a news article or headline:\")\n",
    "\n",
    "if st.button(\"Predict\"):\n",
    "    if user_input.strip() == \"\":\n",
    "        st.warning(\"Please enter some text.\")\n",
    "    else:\n",
    "        cleaned = clean_text(user_input)\n",
    "        vectorized = vectorizer.transform([cleaned])\n",
    "        prediction = model.predict(vectorized)[0]\n",
    "\n",
    "        if prediction == 1:\n",
    "            st.success(\"✅ This looks like *Real News*.\")\n",
    "        else:\n",
    "            st.error(\"⚠️ This looks like *Fake News*.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "16a70f8a-774c-42b4-ac64-55121eb4b893",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e260c07f-ddde-4137-9234-2e4f37c7bda8",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
