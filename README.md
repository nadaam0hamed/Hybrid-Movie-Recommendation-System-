# 🎬 Hybrid Movie Recommendation System

A modern Hybrid Movie Recommendation System built with Streamlit that combines:

- Content-Based Filtering
- Collaborative Filtering
- Hybrid Recommendation Techniques

The application provides personalized movie recommendations using movie similarity and user rating behavior.

---

# 🚀 Features

✅ Hybrid recommendation engine  
✅ Content-based filtering using TF-IDF + Cosine Similarity  
✅ Collaborative filtering using matrix factorization  
✅ Interactive Streamlit UI  
✅ Adjustable hybrid weights  
✅ Evaluation metrics dashboard  
✅ Animated modern interface  
✅ Responsive movie cards with hover effects  
✅ Dataset statistics section  

---

# 🧠 Recommendation Techniques

## 1. Content-Based Filtering

Recommends movies similar to the user's favorite movie using:

- TF-IDF Vectorization
- Cosine Similarity
- Movie genres and tags

---

## 2. Collaborative Filtering

Predicts user preferences using:

- User ratings
- Matrix factorization / collaborative learning

Evaluation metrics include:

- RMSE
- MAE
- Precision
- Recall
- F1-Score

---

## 3. Hybrid Recommendation

Combines both systems:

Final Score =

```python
(weight_content × content_score)
+
(weight_collab × collaborative_score)