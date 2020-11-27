import json
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

train_data = []
train_labels = []
with open('data/train.jsonl', 'r') as infile:
    for line in infile.readlines():
        entry = json.loads(line)
        train_data.append(entry['response']+' '+' '.join(entry['context']))
        train_labels.append(entry['label'])

text_clf.fit(train_data, train_labels)

test_data = []
test_labels = []
test_ids = []
with open('data/test.jsonl', 'r') as infile:
    for line in infile.readlines():
        entry = json.loads(line)
        test_data.append(entry['response']+' '+' '.join(entry['context']))
        #test_labels.append(entry['label'])
        test_ids.append(entry['id'])

predicted = text_clf.predict(test_data)
with open('answer.txt', 'w') as f:
    for i, id in enumerate(test_ids):
        f.write(id+','+predicted[i]+'\n')

#print(np.mean(predicted == twenty_test.target))
