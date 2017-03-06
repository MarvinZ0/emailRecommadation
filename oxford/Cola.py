import heapq
import random
import operator
import pandas as pd
from collections import Counter
from TextClean import textClean

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
import numpy as np
from scipy import spatial
from scipy.io import mmwrite
import csv

def load_file(path_to_data,set_filename,info_filename):
    set = pd.read_csv(path_to_data + set_filename, sep=',', header=0)
    info = pd.read_csv(path_to_data + info_filename, sep=',', header=0)
    return set,info



def get_emails_ids_per_sender(training):

    emails_ids_per_sender = {}
    for index, series in training.iterrows():
        row = series.tolist()
        sender = row[0]
        ids = row[1:][0].split(' ')
        emails_ids_per_sender[sender] = [int(x) for x in ids]

    return emails_ids_per_sender

def get_address_book(emails_ids_per_sender, training_info):
    address_books = {}

    for sender, ids in emails_ids_per_sender.iteritems():
        recs_temp = []
        for my_id in ids:
            recipients = training_info[training_info['mid'] == int(my_id)]['recipients'].tolist()
            recipients = recipients[0].split(' ')
            # keep only legitimate email addresses
            recipients = [rec for rec in recipients if '@' in rec]
            recs_temp.append(recipients)
        # flatten
        recs_temp = [elt for sublist in recs_temp for elt in sublist]
        # compute recipient counts
        rec_occ = dict(Counter(recs_temp))
        # order by frequency
        sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse=True)
        # save
        # address_books[sender] = sorted_rec_occ
        # address_books[sender] = sorted_rec_occ[:100]
        address_books[sender] = [x[0] for x in sorted_rec_occ[0:11]]

    return address_books

def get_unique_user(all_senders,all_recs):
    all_users = []
    all_users.extend(all_senders)
    all_users.extend(all_recs)
    all_users = list(set(all_users))
    return all_users

def result_to_csv(path_to_results,predictions_per_sender):
    # # path_to_results = # fill me!
    # path_to_results = "../results/"

    with open(path_to_results + 'predictions_random.txt', 'wb') as my_file:
        my_file.write('mid,recipients' + '\n')
        for sender, preds in predictions_per_sender.iteritems():
            ids = preds[0]
            random_preds = preds[1]
            for index, my_preds in enumerate(random_preds):
                my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')

    with open(path_to_results + 'predictions_frequency.txt', 'wb') as my_file:
        my_file.write('mid,recipients' + '\n')
        for sender, preds in predictions_per_sender.iteritems():
            ids = preds[0]
            freq_preds = preds[2]
            for index, my_preds in enumerate(freq_preds):
                my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')


if __name__ == "__main__":

    train_filename = '../data/merged_textclean_train.csv'
    train = pd.read_csv(train_filename, sep=',', header=0)
    test_filename = '../data/test_info.csv'

    training_set_filename = '../data/training_set.csv'
    training_set = pd.read_csv(training_set_filename, sep=',', header=0)


    # # dict {sender:[(str)mid1,mid2,mid3...]...}
    emails_ids_per_sender = get_emails_ids_per_sender(training_set)

    # # save all unique sender names list [(str)sender1,sender2...]
    all_senders = emails_ids_per_sender.keys()
    #
    # # create address book with frequency information for each user dict{(str)sender1:[((str)rec1,freq1),(rec2,freq)...]...}
    address_books = {}
    for sender in all_senders:
        book = train[train.sender==sender]['recipients'].tolist()
        book = [x for subitem in book for x in subitem.split(" ")]
        book = list(set(book))
        address_books[sender] = book

    contact_book = {}




