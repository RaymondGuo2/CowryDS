import nltk
from nlp import obtain_corpus, normalise_corpus, build_feature_matrix, get_topics_terms_weights, print_topics_udf
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sentiment import analyse_sentiment_textblob
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

df_control = pd.read_excel('../data_source/CDS_25_Task2.xlsx', 'C Control')
general_corpus = obtain_corpus(df_control)
norm_corpus = normalise_corpus(general_corpus)

vectoriser, tfidf_matrix = build_feature_matrix(norm_corpus, feature_type='tfidf')

total_topics = 2
lda = LatentDirichletAllocation(n_components=total_topics, max_iter=100, learning_method='online', learning_offset=50., random_state=42)
lda.fit(tfidf_matrix)

feature_names = vectoriser.get_feature_names_out()
weights = lda.components_

topics = get_topics_terms_weights(weights, feature_names)
print_topics_udf(topics=topics, total_topics=total_topics, num_terms=8, display_weights=True)


# Sentiment Analysis

sentiment_results = [analyse_sentiment_textblob(doc, verbose=False) for doc in general_corpus]
sentiment_labels, polarities, subjectivities = zip(*sentiment_results)
sentiment_df = pd.DataFrame({
    'document': general_corpus,
    'sentiment_label': sentiment_labels,
    'polarity': polarities,
    'subjectivity': subjectivities
})
print(sentiment_df.head())
sentiment_df.to_csv('../data_source/sentiment_analysis_results.csv', index=False)

# Train classification model
topic_distributions = lda.transform(tfidf_matrix)
print(topic_distributions)

binary_labels = [1 if label == 'positive' else 0 for label in sentiment_labels]
X_train, X_test, y_train, y_test = train_test_split(topic_distributions, binary_labels, test_size=0.2, random_state=42)

clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))