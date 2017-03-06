import heapq

from sklearn.feature_extraction.text import TfidfVectorizer
from oxford.TextClean import TextClean
from numpy.linalg import norm
import numpy as np
import pandas as pd
from collections import Counter
import operator
from sklearn.feature_extraction.text import CountVectorizer
import re
from scipy.spatial.distance import cosine


di = {'a':1.1,'b':3.3,'c':2.1}
da = {'b':1,'c':2,'a':3}
rank = {}
# for k in di.iterkeys():
#         rank[k] = 0.6/di[k] + 0.4/da[k]
#
# sorted_rank = [x[0] for x in sorted(rank.items(), key=operator.itemgetter(1),reverse=True)][:10]
# print sorted_rank

# sorted_rank = sorted(di.items(), key=operator.itemgetter(1),reverse=True)
# r = 1
# re = {}
# for i in xrange(len(sorted_rank)):
#     re[sorted_rank[i][0]] = r
#     r += 1
#
# print re
#
# print type({key: rank for rank, key in enumerate(sorted(di, key=di.get, reverse=True), 1)})
lst = [1,2,3,4,999]
print [min(lst) if x==999 else x for x in lst]
ist = 1
