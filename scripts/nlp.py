# Package for NLP functions

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import html
import re
import nltk
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import numpy as np
import os
from sklearn.decomposition import LatentDirichletAllocation
from sentence_transformers import SentenceTransformer, util
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
# import pyLDAvis
# import pyLDAvis.sklearn


stopword_list = nltk.corpus.stopwords.words('english')
wnl = WordNetLemmatizer()

CONTRACTION_MAP = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "I'd": "I would",
    "I'll": "I will",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "'s": " is",
    "'re": " are",
    "'ll": " will",
    "'ve": " have",
    "'d": " would",
    "n't": " not"
}

def obtain_corpus(df):
    raw_corpus = df['LTR_COMMENT'].tolist()
    return raw_corpus

def unescape_html(parser, text):
    return parser.unescape(text)

def tokenise_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def pos_tag_text(text):
    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
        
    tokens = nltk.word_tokenize(text)
    tagged_text = nltk.pos_tag(tokens, tagset='universal')
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(tag)) for word, tag in tagged_text]
    return tagged_lower_text

def lemmatise_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatised_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag else word for word, pos_tag in pos_tagged_text]
    lemmatised_text = ' '.join(lemmatised_tokens)
    return lemmatised_text

def remove_special_characters(text):
    tokens = tokenise_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_stopwords(text):
    tokens = tokenise_text(text)
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def expand_contractions(text, contraction_mapping):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), flags = re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = CONTRACTION_MAP.get(match)
        if not expanded_contraction:
            expanded_contraction = CONTRACTION_MAP.get(match.lower())
        if not expanded_contraction:
            return match 
        return first_char + expanded_contraction[1:]
    
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text



def normalise_corpus(corpus, lemmatise=True, tokenise=False):
    normalised_corpus = []
    for text in corpus:
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        text = html.unescape(text)
        text = expand_contractions(text, CONTRACTION_MAP)
        if lemmatise:
            text = lemmatise_text(text)
        else:
            text = text.lower()
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        if tokenise:
            text = tokenise_text(text)
            normalised_corpus.append(text)
        else:
            normalised_corpus.append(text)
    
    return normalised_corpus


def build_feature_matrix(documents, feature_type='frequency'):
    feature_type = feature_type.lower().strip()
    if feature_type == 'binary':
        vectoriser = CountVectorizer(binary=True, min_df=1, ngram_range=(1,1))
    elif feature_type == 'frequency':
        vectoriser = CountVectorizer(binary=False, min_df=1, ngram_range=(1,1))
    elif feature_type == 'tfidf':
        vectoriser = TfidfVectorizer(min_df=1, ngram_range=(1,1))
    else:
        raise Exception("Wrong feature type entered")
    
    feature_matrix = vectoriser.fit_transform(documents).astype(float)
    return vectoriser, feature_matrix

def get_topics_terms_weights(weights, feature_names):
    feature_names = np.array(feature_names)
    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])
    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights,sorted_indices)])
    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])
    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]
    return topics


def print_topics_udf(topics, total_topics=1, weight_threshold=0.0001, display_weights=False, num_terms=None):
    for index in range(total_topics):
        topic = topics[index]
        topic = [(term, float(wt)) for term, wt in topic]
        topic = [(word, round(wt,2))for word, wt in topic if abs(wt) >= weight_threshold]
        if display_weights:
            print('Topic #'+ str(index+1) +' with weights')
            print(topic[:num_terms] if num_terms else topic)
        else:
            print('Topic #'+str(index+1)+' without weights')
            tw = [term for term, wt in topic]
            print(tw[:num_terms] if num_terms else tw)
    print

def lda_topicmodel(corpus, num_topics):
    vectoriser, tfidf_matrix = build_feature_matrix(corpus, feature_type='tfidf')
    total_topics = num_topics
    lda = LatentDirichletAllocation(n_components=total_topics, max_iter=100, learning_method='online', learning_offset=50., random_state=42)
    lda.fit(tfidf_matrix)
    feature_names = vectoriser.get_feature_names_out()
    weights = lda.components_
    topics = get_topics_terms_weights(weights, feature_names)
    print_topics_udf(topics=topics, total_topics=total_topics, num_terms=8, display_weights=True)
    return lda, tfidf_matrix, feature_names
    # lda_vis = pyLDAvis.sklearn.prepare(lda, tfidf_matrix, vectoriser)
    # pyLDAvis.save_html(lda_vis, 'lda_visualization.html')

def match_themes_from_corpus(corpus, model, themes, theme_embeddings):

    comment_embeddings = model.encode(corpus, convert_to_tensor=True)
    cos_scores = util.cos_sim(comment_embeddings, theme_embeddings)

    results = []
    for idx, row in enumerate(cos_scores):
        best_match_idx = row.argmax().item()
        best_score = row[best_match_idx].item()
        matched_theme = themes[best_match_idx]

        results.append({
            "comment": corpus[idx],
            "matched_theme": matched_theme,
            "confidence": round(best_score, 3)
        })

    df_results = pd.DataFrame(results)
    theme_percentages = df_results['matched_theme'].value_counts(normalize=True) * 100

    return df_results, theme_percentages

def plot_all_topics_grid(lda_model, feature_names, n_top_words=8, cols=2, figsize=(12, 10)):
    weights = lda_model.components_
    n_topics = weights.shape[0]
    rows = math.ceil(n_topics / cols)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for topic_idx, topic_weights in enumerate(weights):
        top_indices = topic_weights.argsort()[::-1][:n_top_words]
        top_terms = [feature_names[i] for i in top_indices]
        top_weights = [topic_weights[i] for i in top_indices]

        ax = axes[topic_idx]
        sns.barplot(x=top_weights, y=top_terms, ax=ax, palette='viridis')
        ax.set_title(f"Topic {topic_idx}")
        ax.set_xlabel("")
        ax.set_ylabel("")
    
    # Remove any unused subplots
    for i in range(n_topics, len(axes)):
        fig.delaxes(axes[i])

    fig.suptitle("Top Words per Topic", fontsize=16)
    plt.show()