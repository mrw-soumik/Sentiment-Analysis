# Sentiment Analysis Project

## Project Overview
This project is a **Sentiment Analysis Tool** designed to classify text into three sentiment categories: **positive**, **neutral**, and **negative**. It leverages Natural Language Processing (NLP) for text preprocessing and machine learning models to classify text based on sentiment. The tool is useful for understanding opinions in text data, such as customer feedback or social media posts.

## Technologies Used
- **Python**: Programming language for building and running the project.
- **NLTK**: Used for tokenization, stopword removal, and other NLP preprocessing steps.
- **Scikit-learn**: Machine learning library used for vectorization (TF-IDF), model building (Naive Bayes, Logistic Regression, SVM), and evaluation.
- **Pandas**: Data manipulation library to handle text data.

## Dataset
The dataset, `sentiment_dataset.txt`, contains 300 labeled text samples:
- **100 Positive Samples**: Reflect positive opinions.
- **100 Neutral Samples**: Reflect neutral or mixed opinions.
- **100 Negative Samples**: Reflect negative opinions.

Each line in the dataset contains a text sample and its corresponding label (positive, neutral, or negative), separated by two spaces.

Example format:
```plaintext
I absolutely love this product! It works great.    positive
The product is average, not bad but not great.     neutral
This is the worst purchase I’ve made.              negative
```

## Project Structure
- **sentiment_analysis.py**: Main script that preprocesses data, trains models, evaluates performance, and predicts sentiment for a sample text.
- **sentiment_dataset.txt**: Contains labeled sentiment data used for training and testing.
- **README.md**: Instructions for setup, running, and using the project.

## Installation

### Requirements
Ensure you have Python 3.6+ and install the required libraries:
```bash
pip install nltk scikit-learn pandas
```

### NLTK Data
The script automatically downloads required NLTK data (e.g., `punkt`, `stopwords`). Ensure you’re connected to the internet for the first run.

## Running the Project

1. **Dataset**: Ensure `sentiment_dataset.txt` is in the same directory as `sentiment_analysis.py`.
2. **Run the Script**:
   ```bash
   python sentiment_analysis.py
   ```

The script performs the following tasks:
- **Preprocesses** text by removing punctuation, converting to lowercase, removing stopwords, and tokenizing.
- **Splits** the data into training and test sets (80/20 split).
- **Vectorizes** text using TF-IDF to convert text data into numerical form.
- **Trains** three machine learning models: Naive Bayes, Logistic Regression, and SVM.
- **Evaluates** each model's accuracy and generates a classification report.

### Example Output
After running the script, you’ll see accuracy scores and detailed classification reports for each model. An example sentiment prediction is also displayed using the best-performing model.

## Usage

To predict the sentiment of a custom text, use the `predict_sentiment` function in `sentiment_analysis.py`:

```python
print("Predicted Sentiment:", predict_sentiment("The product exceeded my expectations!"))
```

Replace `"The product exceeded my expectations!"` with your own text.

## License
This project is open-source and can be freely used and modified for personal or educational purposes.
