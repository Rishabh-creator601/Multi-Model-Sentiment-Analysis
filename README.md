![Image](https://github.com/user-attachments/assets/bf33ee06-2e6d-4831-9d37-1db0b427186a)


VISIT DEPLOYED APP : [HERE](https://multi-model-sentiment.streamlit.app/)
KAGGLE CODE :  [HERE](https://www.kaggle.com/code/rishabh2007/customer-feedback-sentiment-analysis-95)
MEDIUM ARTICLE : [Here](https://medium.com/h7w/04e9604e27b8)

📌 Project Description

This project is a Multi-Model Sentiment Evaluation App built using Streamlit.
It allows users to analyze the sentiment of a given text using multiple sentiment analysis approaches at once.

Instead of relying on a single model, the app compares predictions from:
Rule-based models (VADER)
Statistical models (TextBlob)
Machine Learning models (Gaussian Naive Bayes, Random Forest, XGBoost)
All model outputs are shown together, and a final sentiment decision is generated based on the combined result.
This project is mainly focused on learning, comparison, and practical implementation of NLP sentiment techniques.

🔄 Application Flow

User Input

User can either type a sentence or select one of the example texts shown as pills.

Text Processing

The input text is passed to different sentiment models without modifying the original text for rule-based models.

For ML models, the text is cleaned and vectorized using TF-IDF.

Model Predictions

VADER analyzes polarity scores
TextBlob calculates sentiment polarity
Naive Bayes, Random Forest, and XGBoost predict sentiment using trained models
Result Display
Each model returns a binary sentiment score (Positive or Negative).
All results are displayed in a table for easy comparison.
Final Decision
The average of all model predictions is calculated.
Based on this average, the app shows the Overall Review as Positive or Negative.


🛠 Tech Stack

- Python
- Streamlit
- NLTK (VADER)
- TextBlob
- Scikit-learn

XGBoost

TF-IDF Vectorization
