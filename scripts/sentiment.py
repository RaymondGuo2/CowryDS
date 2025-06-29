# Create Python package for sentiment analysis

import pandas as pd
from textblob import TextBlob
from normalise import normalise_accented_characters, strip_html
import html

# Define sentiment analysis function
def analyse_sentiment_textblob(review, threshold=0.1, verbose=False):
    # Normalise and clean input
    review = normalise_accented_characters(review)
    review = html.unescape(review)
    review = strip_html(review)

    # Analyse sentiment
    blob = TextBlob(review)
    sentiment_score = round(blob.sentiment.polarity, 2)
    sentiment_subjectivity = round(blob.sentiment.subjectivity, 2)
    final_sentiment = 'positive' if sentiment_score >= threshold else 'negative'

    if verbose:
        sentiment_frame = pd.DataFrame(
            [[final_sentiment, sentiment_score, sentiment_subjectivity]],
            columns=pd.MultiIndex.from_tuples([
                ('SENTIMENT STATS:', 'Predicted Sentiment'),
                ('SENTIMENT STATS:', 'Polarity Score'),
                ('SENTIMENT STATS:', 'Subjectivity Score')
            ])
        )
        print(sentiment_frame)

    return final_sentiment, sentiment_score, sentiment_subjectivity