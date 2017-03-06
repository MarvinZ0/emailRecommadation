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
    path_to_data = '../data/Removed0001_0002/'
    set_filename = "training_set_Removed0001_startsFrom_2001.csv"
    info_filename = "training_info_Removed0001_startsFrom_2001.csv"

    ##########################
    # load some of the files #
    ##########################

    training, training_info = load_file(path_to_data,set_filename,info_filename)

    ################################
    # create some handy structures #
    ################################

    # dict {sender:[(str)mid1,mid2,mid3...]...}
    emails_ids_per_sender = get_emails_ids_per_sender(training)

    # save all unique sender names list [(str)sender1,sender2...]
    all_senders = emails_ids_per_sender.keys()

    # create address book with frequency information for each user dict{(str)sender1:[((str)rec1,freq1),(rec2,freq)...]...}
    address_books = get_address_book(emails_ids_per_sender, training_info)

    # save all unique recipient names
    # all_recs = list(set([elt[0] for sublist in address_books.values() for elt in sublist]))

    # save all unique user names
    # all_users = get_unique_user(all_senders,all_recs)


    test = pd.read_csv('../data/' + 'test_set.csv', sep=',', header=0)
    test_info = pd.read_csv('../data/' + 'test_info.csv', sep=',', header=0)







    #############
    # baselines #
    #############

    # # will contain email ids, predictions for random baseline, and predictions for frequency baseline
    # predictions_per_sender = {}
    #
    # # number of recipients to predict
    # k = 10
    #
    # for index, row in test.iterrows():
    #     name_ids = row.tolist()
    #     sender = name_ids[0]
    #     # get IDs of the emails for which recipient prediction is needed
    #     ids_predict = name_ids[1].split(' ')
    #     ids_predict = [int(my_id) for my_id in ids_predict]
    #     random_preds = []
    #     freq_preds = []
    #     # select k most frequent recipients for the user
    #     k_most = [elt[0] for elt in address_books[sender][:k]]
    #     for id_predict in ids_predict:
    #         # select k users at random
    #         random_preds.append(random.sample(all_users, k))
    #         # for the frequency baseline, the predictions are always the same
    #         freq_preds.append(k_most)
    #     predictions_per_sender[sender] = [ids_predict, random_preds, freq_preds]



    ##############
    # My own way #
    ##############
    # will contain email ids, predictions for random baseline, and predictions for frequency baseline
    predictions_per_sender = {}

    vecto_sender_rec = {}
    # path_to_vocab = '../data/vocab_senders/'
    # for index, sender in enumerate(all_senders):
    #     vectorizer = CountVectorizer(max_features=5000)
    #     bodies = training_info[training_info['mid'].isin(emails_ids_per_sender[sender])]['body'].tolist()
    #     for index in xrange(len(bodies)):
    #         bodies[index] = textClean(bodies[index])
    #     vectorizer.fit(bodies)
    #     vecto_sender[sender] = vectorizer
    #
    #     with open(path_to_vocab + sender + '.csv', 'wb') as csvfile:
    #         cwriter = csv.writer(csvfile, delimiter=' ')
    #         for word in vectorizer.get_feature_names():
    #             cwriter.writerow([word])
    #     csvfile.close()
    #
    # sender_recs_matrix = []
    #
    # for index, sender in enumerate(all_senders):
    #     sender_mails = training[training[sender] == sender]['mids'].tolist()
    #     sender_mails = sender_mails[0].split()
    #     sender_mails = training_info[training_info['mids'].isin(sender_mails)]['body','recipients']
    #     for rec in address_books[sender]:
    #         rec_sender_mail = sender_mails[sender_mails['recipients'].__contains__(rec)==True]['body']
    #         rec_sender_vecto = vecto_sender[sender].transform(rec_sender_mail)
    #         # rec_sender_vecto =

    i = 0
    for index, row in test.iterrows():
        name_ids = row.tolist()
        name = name_ids[0]
        ids = name_ids[1].split(' ')
        preds = []
        for mid in ids:
            vectorizer = CountVectorizer(max_features=5000)
            bodies = training_info[training_info['mid'].isin(emails_ids_per_sender[name])]['body'].tolist()
            bodies = [textClean(x) for x in bodies]
            vectorizer.fit(bodies)

            rec_vect = {}
            if not vecto_sender_rec.has_key(name):
                rec_d = {}
                for rec in address_books[name]:
                    rec_sender_mail = training_info[training_info['recipients'].str.contains(rec)]['body'].tolist()
                    rec_sender_mail = [textClean(x) for x in rec_sender_mail]
                    temp = vectorizer.transform(rec_sender_mail).toarray()
                    if len(rec_sender_mail)>1:
                        linfnorm = norm(temp, axis=1, ord=1)
                        temp = temp.astype(np.float) / linfnorm[:, None]
                        temp = np.sum(temp, axis=0)
                    linfnorm = norm(temp, axis=0, ord=1)
                    temp = temp.astype(np.float) / linfnorm
                    rec_d.update({rec:temp})
                vecto_sender_rec[name] = rec_d

            test_mail = test_info[test_info['mid'] == int(mid)]['body'].tolist()
            test_mail = [textClean(x) for x in test_mail]

            vecto_new = vectorizer.transform(test_mail).toarray()
            linfnorm = norm(vecto_new, axis=0, ord=np.inf)
            vecto_new = vecto_new.astype(np.float) / linfnorm

            cosine_similarity = []

            for rec in address_books[name]:
                vecto_rec = vecto_sender_rec[name][rec]
                vecto_rec = np.nan_to_num(vecto_rec)
                vecto_new = np.nan_to_num(vecto_new)
                c_s = 1 - spatial.distance.cosine(vecto_rec, vecto_new)
                cosine_similarity.append(c_s)

            top_10 = zip(*heapq.nlargest(10, enumerate(cosine_similarity), key=operator.itemgetter(1)))[0]
            c_s_preds = []

            for nu in top_10:
                c_s_preds.append(address_books[name][nu])
            preds.append(c_s_preds)
        predictions_per_sender[name] = [ids, preds]

        i += 1
        print 'Done %d prediction'%i

    with open('../results/ownway/' + 'predictions_c_s_startsFrom2001.txt', 'wb') as my_file:
        my_file.write('mid,recipients' + '\n')
        for sender, preds in predictions_per_sender.iteritems():
            ids = preds[0]
            c_s_preds = preds[1]
            for index, my_preds in enumerate(c_s_preds):
                my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')








    #################################################
    # write predictions in proper format for Kaggle #
    #################################################

    # path_to_results = "../results/"
    # result_to_csv(path_to_results,predictions_per_sender)
