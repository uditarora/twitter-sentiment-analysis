import pandas as pd
import numpy as np
from time import time
import joblib
import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from cleaner import TextCleaner
from twitterclient import TwitterClient

model_filename = "saved_model.pkl"

def train(eval=True):
    dataset = pd.read_csv('dataset/airline_tweets.csv')
    # dataset = pd.read_csv("dataset/full-corpus.csv")
    
    dataset = dataset[['text', 'airline_sentiment']]
    # print(str(dataset.airline_sentiment.value_counts()))

    tc = TextCleaner()
    dataset.text = tc.fit_transform(dataset.text)

    cv = CountVectorizer(max_df=0.5, ngram_range=(1,3))

    # print(dataset.columns.values)

    counts = cv.fit_transform(dataset.text)

    X_train, X_test, y_train, y_test = train_test_split(counts, dataset.airline_sentiment, test_size=0.1, random_state=37)

    X_train = X_train.todense()
    X_test = X_test.todense()

    # print(X_train.shape, y_train.shape)
    # print(X_test.shape, y_test.shape)

    # param_grid = {
    #     'alpha': [0.4, 0.5, 0.6]
    # }

    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'class_weight': ['balanced', None],
        'multi_class': ['ovr', 'multinomial']
    }

    # clf = GridSearchCV(MultinomialNB(), param_grid, verbose=10)
    # clf = GridSearchCV(LogisticRegression(), param_grid, verbose=10)

    clf = LogisticRegression(C=1.0)
    # clf = MultinomialNB(alpha=0.5)

    print("\nFitting the classifier to the training set")
    t0 = time()

    clf.fit(X_train, y_train)

    print("done in %0.3fs" % (time() - t0))

    # Save the model
    model = {}
    model['vec'] = cv
    model['clf'] = clf
    joblib.dump(model, model_filename)

    # print("Best estimator found by grid search:")
    # print(clf.best_estimator_)

    if eval == False:
        return model

    print("\nCalculating evaluation metrics")
    t0 = time()
    print("Train accuracy:")
    print(str(clf.score(X_train, y_train)))
    print("Test accuracy:")
    print(str(clf.score(X_test, y_test)))

    pred = clf.predict(X_test)

    print(classification_report(y_test, pred))

    print("done in %0.3fs" % (time() - t0))

    return model

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract sentiment from tweets')
    parser.add_argument('--tweets', dest='tweets_file', default=None,
                       help='The path of the file containing tweets')
    parser.add_argument('--query', dest='query', default='#RoyalWedding',
                       help='The twitter query')
    parser.add_argument("--train", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Force the model to retrain")
    parser.add_argument("--eval", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Force the model to retrain")

    args = parser.parse_args()

    if args.train:
        print("Forcing the model to retrain")
        model = train(args.eval)
    else:
        try:
            model = joblib.load(model_filename)
        except:
            model = train(args.eval)

    cv = model['vec']
    clf = model['clf']

    if args.tweets_file is not None:
        with open(args.tweets_file, 'r') as f:
            tweets = f.read().split('\n')
            df = pd.DataFrame(tweets, columns=['text'])
    else:
        client = TwitterClient()
        tweets = client.get_tweets(query = args.query, count = 200)
        df = pd.DataFrame(tweets)

    tc = TextCleaner()
    cleaned_text = tc.fit_transform(df.text)

    counts = cv.transform(cleaned_text)
    preds = clf.predict(counts)

    for text, pred in zip(df.text, preds):
        print("\nSentiment: %s. Tweet: %s" % (pred, text))
