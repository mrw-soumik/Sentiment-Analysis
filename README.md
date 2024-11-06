
# Sentiment Analysis Tool

**Technologies:** Python, NLTK, Scikit-learn  

## Objective
This project builds a sentiment analysis tool that classifies text as positive, negative, or neutral, which can be useful for analyzing opinions in social media, reviews, and other text sources.

## Key Contributions
- Preprocessed data using tokenization, stemming, and removal of stopwords.
- Built a Naive Bayes classifier using Scikit-learn to predict sentiment based on labeled text data.
- Evaluated the model’s performance to assess its effectiveness in sentiment classification.

## Outcome
The sentiment analysis tool accurately classifies sentiments for text data, showcasing practical applications in opinion mining.

## Installation and Setup

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/yourusername/Sentiment-Analysis-Project.git
    cd Sentiment-Analysis-Project
    ```

2. **Install Required Packages**:
    Install the necessary Python packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Model**:
    Execute `sentiment_analysis.py` to train the model and see the output.
    ```bash
    python sentiment_analysis.py
    ```

## Usage
To analyze new text:
1. Use the `predict_sentiment` function in `sentiment_analysis.py`.
2. Pass any custom text to classify it as positive, negative, or neutral.

Example:
```python
print(predict_sentiment("I absolutely love this product!"))
```

## License
This project is licensed under the MIT License.
