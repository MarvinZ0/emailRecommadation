import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy
import pylab
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import spatial
from pylab import rcParams
rcParams['figure.figsize'] = 20, 5



def score_recency(sender, t_new, address_books, contact_books, merge_train):
    score = {}
    for rec in address_books[sender]:
        rcy = 0
        for mid in contact_books[sender][rec]:
            t_ancien = merge_train[merge_train.mid == mid]['date'].tolist()[0]
            rcy += recency(t_new, t_ancien)
        rcy = 6 * rcy

        if contact_books.has_key(rec) and contact_books[rec].has_key(sender):
            for mid in contact_books[rec][sender]:
                t_ancien = merge_train[merge_train.mid == mid]['date'].tolist()[0]
                rcy += recency(t_new, t_ancien)
        score[rec] = rcy
    return score


def recency(t_new, t_ancien):
    t_new = numpy.datetime64(t_new).astype("datetime64[D]")
    t_ancien = numpy.datetime64(t_ancien).astype("datetime64[D]")
    s = (t_new - t_ancien).astype("int") ** (-1.5)
    return s





def score_content(sender ,address_books ,contact_books ,merge_train ,vectorizer = TfidfVectorizer()):
    score = {}
    for rec in address_books[sender]:
        cnt = 0
        for mid in contact_books[sender][rec]:

            pass

def content_similarity():
    pass






def get_emails_ids_per_sender(training):

    emails_ids_per_sender = {}
    for index, series in training.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        emails_ids_per_sender[sender] = [int(x) for x in ids]

    return emails_ids_per_sender

training_set_filename = '../data/training_set.csv'
training_set = pd.read_csv(training_set_filename, sep=',', header=0)
emails_ids_per_sender = get_emails_ids_per_sender(training_set)


all_senders = emails_ids_per_sender.keys()
train_filename = '../data/merged_textclean_train.csv'
train = pd.read_csv(train_filename, sep=',', header=0)

address_books = {}
for sender in all_senders:
    book = train[train.sender==sender]['recipients'].tolist()
    book = [x for subitem in book for x in subitem.split(" ")]
    book = list(set(book))
    address_books[sender] = book

contact_books = {}
for sender in all_senders:
    mails = train[train.sender==sender]
    book = {}
    for rec in address_books[sender]:
        rec_mail =mails[mails.recipients.str.contains(rec)]['mid'].tolist()
        book[rec] = rec_mail
    contact_books[sender] = book








test_filename = '../data/merged_textclean_test.csv'
test = pd.read_csv(test_filename, sep=',', header=0)

merged_train_filename = '../data/merged_textclean_train.csv'
train = pd.read_csv(merged_train_filename, sep=',', header=0)

cnt_vectorizer = TfidfVectorizer(max_features=5000)
cnt_vectorizer.fit(train.body.values.astype(str))

pred_mid = {}
i = 0
for index, serie in test.iterrows():
    row = serie.tolist()
    m_id = row[0]
    sender = row[1]
    timestamp = row[2]
    body = row[3]
    rcy_score = score_recency(sender, timestamp, address_books, contact_books, train)

    cnt_score = {}
    vect_new = cnt_vectorizer.transform(([body] if type(body) == str else [body.astype(str)])).toarray()
    vect_new = numpy.round_(vect_new, 3)
    if numpy.isnan(vect_new).any() or numpy.isinf(vect_new).any():
        vect_new = numpy.nan_to_num(vect_new)
    # vect_new = numpy.nan_to_num(vect_new)
    vect_new = vect_new.astype(numpy.float) / (numpy.sum(vect_new) if not numpy.sum(vect_new) == 0 else 1.0)
    vect_new = numpy.round_(vect_new, 3)
    if numpy.isnan(vect_new).any() or numpy.isinf(vect_new).any():
        vect_new = numpy.nan_to_num(vect_new)
    for rec in address_books[sender]:
        cnt = 0
        for mid in contact_books[sender][rec]:
            cont = train[train.mid == mid]['body'].values[0]
            cont = (cont if type(cont) == str else cont.astype(str))
            vect = cnt_vectorizer.transform([cont]).toarray()
            vect = numpy.round_(vect, 3)
            #             vect = numpy.nan_to_num(vect)
            if numpy.isnan(vect).any() or numpy.isinf(vect).any():
                vect = numpy.nan_to_num(vect)
            vect = vect.astype(numpy.float) / (numpy.sum(vect) if not numpy.sum(vect) == 0 else 1.0)
            vect = numpy.round_(vect, 3)
            if numpy.isnan(vect).any() or numpy.isinf(vect).any():
                vect = numpy.nan_to_num(vect)
            # vect = numpy.nan_to_num(vect)
            c_s = 1 - spatial.distance.cosine(vect_new, vect)
            cnt += c_s
        cnt = 6 * cnt

        if contact_books.has_key(rec) and contact_books[rec].has_key(sender):
            for mid in contact_books[rec][sender]:
                conte = train[train.mid == mid]['body'].values[0]
                conte = (conte if type(conte) == str else conte.astype(str))
                vect = cnt_vectorizer.transform([conte]).toarray()
                vect = numpy.round_(vect, 3)
                if numpy.isnan(vect).any() or numpy.isinf(vect).any():
                    vect = numpy.nan_to_num(vect)
                vect = vect.astype(numpy.float) / (numpy.sum(vect) if not numpy.sum(vect) == 0 else 1.0)
                vect = numpy.round_(vect, 3)
                if numpy.isnan(vect).any() or numpy.isinf(vect).any():
                    vect = numpy.nan_to_num(vect)
                # vect = numpy.nan_to_num(vect)
                c_s = 1 - spatial.distance.cosine(vect_new, vect)
                cnt += c_s
        cnt_score[rec] = cnt

    rcy_score = {k: 0.4 / v for k, v in rcy_score.iteritems()}
    cnt_score = {k: 0.6 / v for k, v in cnt_score.iteritems()}

    top = [x[0] for x in (Counter(rcy_score) + Counter(cnt_score)).most_common(10)]
    top = " ".join(top)

    pred_mid[m_id] = top
    i += 1
    if i % 100 == 0:
        print i






pred_mid_test = pred_mid

merged_test_filename = '../data/merged_textclean_test.csv'
test = pd.read_csv(merged_test_filename, sep=',', header=0)
test.dtypes
test['body'].fillna(str(0), inplace=True)
test.to_csv(merged_test_filename, index=False)

df = pd.DataFrame.from_dict(pred_mid_test, orient='index').reset_index()
df.rename(columns={'index': 'mid', 0: 'recipients'}, inplace=True)
print df.head()


df.to_csv('../results/ownway/pcsn.csv', index=False)