import random
import operator
import pandas as pd
from collections import Counter
from TextClean import textClean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from numpy.linalg import norm
import numpy as np
from scipy.io import mmwrite
import csv



#path_to_data = # fill me!
path_to_data = '../data/'

##########################
# load some of the files #
##########################

training = pd.read_csv(path_to_data + 'training_set.csv', sep=',', header=0)

training_info = pd.read_csv(path_to_data + 'training_info.csv', sep=',', header=0)

test = pd.read_csv(path_to_data + 'test_set.csv', sep=',', header=0)


################################
# create some handy structures #
################################


# convert training set to dictionary
emails_ids_per_sender = {}
for index, series in training.iterrows():
    row = series.tolist()
    sender = row[0]
    ids = row[1:][0].split(' ')
    emails_ids_per_sender[sender] = ids

print "Done converting emails_ids_per_sender."

# save all unique sender names
all_senders = emails_ids_per_sender.keys()

# create address book with frequency information for each user
address_books = {}

for sender, ids in emails_ids_per_sender.iteritems():
    recs_temp = []
    for my_id in ids:
        recipients = training_info[training_info['mid']==int(my_id)]['recipients'].tolist()
        recipients = recipients[0].split(' ')
        # keep only legitimate email addresses
        recipients = [rec for rec in recipients if '@' in rec]
        recs_temp.append(recipients)
    # flatten
    recs_temp = [elt for sublist in recs_temp for elt in sublist]
    # compute recipient counts
    rec_occ = dict(Counter(recs_temp))
    # order by frequency
    sorted_rec_occ = sorted(rec_occ.items(), key=operator.itemgetter(1), reverse = True)
    # save
    address_books[sender] = [x[0] for x in sorted_rec_occ[0:51]]

print "Done create address_books"

# train the tf_idf vectorizer
vectoTF = CountVectorizer(max_features=5000)
messages = training_info['body'].tolist()
index = 0
for message in messages:
    if message.__contains__("-----Original Message-----") or message.__contains__("Forwarded by"):
        messages.pop(index)
    else:
        messages[index] = textClean(message)
        index += 1
vectoTF.fit(messages)

with open(path_to_data + 'CountVecto_bag_words.csv', 'wb') as csvfile:
    cwriter = csv.writer(csvfile, delimiter=' ')
    for word in vectoTF.get_feature_names():
        cwriter.writerow([word])
csvfile.close()

print "Done Vectorizer training."



# csv writer
# with open(path_to_data + 'sender_rec_vect.csv', 'wb') as my_file:
#     my_file.write('mid,recipients' + '\n')
#     for sender, preds in predictions_per_sender.iteritems():
#         ids = preds[0]
#         random_preds = preds[1]
#         for index, my_preds in enumerate(random_preds):
#             my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')





# save all recipients for every sender and recipients' td-idf vector normalized to 1.0
# general_dict = {}
# i = 1
# for sender, ids in emails_ids_per_sender.iteritems():
#     if i < 5 :
#         sub_dict = {}
#         # recs_temp = []
#         for my_id in ids:
#             recipients = training_info[training_info['mid']==int(my_id)]['recipients'].tolist()
#             recipients = recipients[0].split(' ')
#             # keep only legitimate email addresses
#             recipients = [rec for rec in recipients if '@' in rec]
#
#             content = training_info[training_info['mid']==int(my_id)]['body'].tolist()
#             content[0] = textClean(content[0])
#             for rec in recipients:
#                 if rec in address_books[sender]:
#                     newVectSpMetx = vectoTF.transform(content).toarray()
#                     # linfnorm = norm(newVect, axis=1, ord=np.inf)
#                     # newVect = newVect.astype(np.float) / linfnorm[:, None]
#                     newVectSpMetx = newVectSpMetx / np.linalg.norm(newVectSpMetx) if np.linalg.norm(newVectSpMetx)!=0 else 1
#                     if not sub_dict.has_key(rec):
#                         sub_dict[rec] = newVectSpMetx
#                     else:
#                         sub_dict[rec] += newVectSpMetx
#         # for key in sub_dict.keys():
#         #     sub_dict[key] = sub_dict[key].tolist()
#
#         general_dict[sender] = sub_dict
#
#     else:
#         pass
#         # recs_temp.append(recipients)
#     print i
#     i+=1
#
#
# print "Done building dictionary"



# # sava all content info messages = [[mid,content],[mid,content]...]
# messages = []
# for index,series in training_info.iterrows():
#     row = series.tolist()
#     mid = row[0]
#     content = textClean(row[2])
#     messages.append([mid,content])
#
# print "Done saving messages"
#
# # TF-idf vectorizer and update messages[]
# to_tfidf=[]
# for cot in messages:
#     to_tfidf.append(cot[1])
#
# vectoTF_IDF = TfidfVectorizer()
# vectoTF_IDF.fit(to_tfidf)
# tf = vectoTF_IDF.transform(to_tfidf)
# # bag_words = vectoTF_IDF.get_feature_names()
#
# print "Done TF_IDF vectorizer"
#
# # update messages[] to messages = [[mid,vecto],[mid,vecto]...]
# linfnorm = norm(tf.toarray(), axis=1, ord=np.inf)
# tf_array = tf.toarray().astype(np.float) / linfnorm[:,None]
# print type(tf_array[0])
#
# for index in xrange(len(messages)):
#     messages[index][1] = tf_array[index]
#
# print "Done Update messages"


















#################################################
# write predictions in proper format for Kaggle #
#################################################

#path_to_results = # fill me!
# path_to_results = "../results/"

# with open(path_to_results + 'predictions_random.txt', 'wb') as my_file:
#     my_file.write('mid,recipients' + '\n')
#     for sender, preds in predictions_per_sender.iteritems():
#         ids = preds[0]
#         random_preds = preds[1]
#         for index, my_preds in enumerate(random_preds):
#             my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')
#
# with open(path_to_results + 'predictions_frequency.txt', 'wb') as my_file:
#     my_file.write('mid,recipients' + '\n')
#     for sender, preds in predictions_per_sender.iteritems():
#         ids = preds[0]
#         freq_preds = preds[2]
#         for index, my_preds in enumerate(freq_preds):
#             my_file.write(str(ids[index]) + ',' + ' '.join(my_preds) + '\n')