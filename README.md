# Spam-Email-Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

An optimized machine learning system for email spam detection using Natural Language Processing (NLP) and multiple classification algorithms. Achieves 95-99% accuracy in distinguishing spam emails from legitimate (ham) emails.

## 🎯 Features

- **Multiple ML Models**: Compares Naive Bayes, Logistic Regression, and Random Forest classifiers
- **Advanced Text Preprocessing**: Removes URLs, emails, numbers, punctuation, and stopwords
- **TF-IDF Vectorization**: Uses unigrams and bigrams for superior feature extraction
- **Comprehensive Evaluation**: Accuracy, AUC-ROC, confusion matrix, and classification reports
- **Smart Column Detection**: Automatically identifies label and text columns in your dataset
- **Visualization**: Class distribution plots, model comparison charts, and ROC curves
- **Easy Prediction**: Simple function to classify new emails
- **Model Persistence**: Saves trained models for future use

## 📊 Performance

| Model | Accuracy | AUC-ROC |
|-------|----------|---------|
| Naive Bayes | ~97% | ~98% |
| Logistic Regression | ~96% | ~97% |
| Random Forest | ~98% | ~99% |

*Results may vary depending on your dataset*

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
nltk
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-spam-detection.git
cd email-spam-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
```

### Usage

#### Option 1: Google Colab (Recommended)

1. Upload `Email_Spam_Detection.ipynb` to Google Colab
2. Upload your dataset (`spam_ham_dataset.csv`)
3. Run all cells sequentially

#### Option 2: Local Python Script

```bash
python email_spam_detection.py
```

#### Option 3: Custom Prediction

```python
from predict import predict_email

# Load the saved model
text = "Congratulations! You've won $1000. Click here now!"
label, confidence = predict_email(text)
print(f"Prediction: {label}")
print(f"Confidence: {confidence}")
```

## 📁 Project Structure

```
email-spam-detection/
│
├── Email_Spam_Detection.ipynb    # Jupyter notebook (Google Colab ready)
├── email_spam_detection.py       # Python script version
├── requirements.txt               # Package dependencies
├── README.md                      # Project documentation
│
├── data/
│   └── spam_ham_dataset.csv      # Your dataset (not included)
│
├── models/
│   ├── spam_detector_model.pkl   # Trained model (generated)
│   └── tfidf_vectorizer.pkl      # TF-IDF vectorizer (generated)
│
└── outputs/
    ├── class_distribution.png     # Visualizations (generated)
    ├── model_comparison.png
    └── confusion_matrix.png
```

## 📝 Dataset Format

Your CSV file should have at least two columns:

| label | text/message |
|-------|--------------|
| spam  | "URGENT! Claim your prize now..." |
| ham   | "Hi, let's meet for coffee tomorrow" |

**Supported column names:**
- **Label column**: `label`, `class`, `category`, `type`
- **Text column**: `text`, `message`, `email`, `sms`

The system will automatically detect your column names!

## 🔬 Methodology

### 1. Text Preprocessing
- Convert to lowercase
- Remove URLs, email addresses, and numbers
- Remove punctuation and extra whitespace
- Tokenization
- Stopword removal

### 2. Feature Extraction
- **TF-IDF Vectorization**: Converts text to numerical features
- **N-grams**: Unigrams and bigrams (1-2 word combinations)
- **Max Features**: 3000 most important features

### 3. Model Training
Three models are trained and compared:
- **Naive Bayes**: Fast, probabilistic classifier
- **Logistic Regression**: Linear model with regularization
- **Random Forest**: Ensemble method with 100 trees

### 4. Evaluation Metrics
- **Accuracy**: Overall correct predictions
- **Precision**: Spam emails correctly identified
- **Recall**: Percentage of spam caught
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Model's discrimination ability

## 📊 Visualizations

The system generates three key visualizations:

1. **Class Distribution**: Shows balance between spam and ham emails
2. **Model Comparison**: Bar chart comparing model accuracies
3. **ROC Curves**: Compares model performance across thresholds
4. **Confusion Matrix**: Shows true/false positives and negatives

## 🔧 Customization

### Adjust TF-IDF Parameters

```python
tfidf = TfidfVectorizer(
    max_features=5000,      # Increase for more features
    ngram_range=(1, 3),     # Use trigrams
    min_df=2,               # Minimum document frequency
    max_df=0.8              # Maximum document frequency
)
```

### Tune Model Hyperparameters

```python
# Random Forest
model = RandomForestClassifier(
    n_estimators=200,       # More trees
    max_depth=50,           # Tree depth
    min_samples_split=5,    # Minimum samples to split
    random_state=42
)
```

## 🧪 Example Predictions

```python
Test Examples:

1. "Congratulations! You've won $1000..."
   → Prediction: SPAM (99.8% confidence)

2. "Hi John, can we meet tomorrow at 3pm?"
   → Prediction: HAM (98.5% confidence)

3. "URGENT: Your account will be suspended..."
   → Prediction: SPAM (97.2% confidence)

4. "Hey, just checking if you're coming to dinner"
   → Prediction: HAM (96.1% confidence)
```

## 🛠️ Troubleshooting

### NLTK LookupError
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
```

### Memory Issues
Reduce `max_features` in TfidfVectorizer:
```python
tfidf = TfidfVectorizer(max_features=1000)
```

### Imbalanced Dataset
Use SMOTE for oversampling:
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
```

## 📈 Future Improvements

- [ ] Deep Learning models (LSTM, BERT)
- [ ] Real-time email classification API
- [ ] Multi-language support
- [ ] Email header analysis
- [ ] Ensemble model combination
- [ ] Web interface for easy testing
- [ ] Integration with email clients

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



## 👨‍💻 Author

**M.Bilal Sharif**
- GitHub: [mbilalsharif08@gmail.com](https://github.com/MBilalSharif)

---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [NLTK Documentation](https://www.nltk.org/)
- [TF-IDF Explanation](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Email Spam Detection Research](https://scholar.google.com/scholar?q=email+spam+detection)


  url = {https://github.com/yourusername/email-spam-detection}
}
```
