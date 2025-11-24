1.
from collections import Counter
import distutils.spawn
import eli5
import random
from eli5.sklearn import PermutationImportance
import keras
from keras_tuner.tuners import RandomSearch
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pandas.core.frame import DataFrame
import pickle
import re
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import socket
import string
import sys

 #DNN imports
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Input, Embedding, Reshape, MaxPool1D, BatchNormalization, SeparableConv1D
from tensorflow.keras.layers import LSTM, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dropout, Dense
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.preprocessing.text as tf_text
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tld import get_tld
from tld.exceptions import TldDomainNotFound
import tldextract
import urllib
from urllib.parse import urlparse, unquote
from xgboost import XGBClassifier
2
# config files
vocab_size = 128          # Size of the vocabulary (unique tokens)
embedding_dim = 64        # Dimension of embedding vectors
num_epoch = 100           # Number of training epochs
batch_size = 64           # Batch size for training
max_len = 256             # Maximum sequence length
lr_rate = 0.001           # Learning rate for the optimizer
 3 # Set seeds for reproducibility
random_state = 42
np.random.seed(random_state)
random.seed(random_state)
tf.random.set_seed(random_state)
 4 Load CSIC and ECML/PKDD combined
waf_data='CAC_DA_Synthetic_logs_500k_20251023.csv'
waf_data=pd.read_csv(waf_data)
5 # number of samples and features
n_features=waf_data.shape[1]
n_samples =waf_data.shape[0]
print("Number of samples:", n_samples)
print("Number of features:", n_features)
6 # print waf data rows and columns
waf_data
7 #show data types
print(waf_data.dtypes)
8 #adding host path and query to get url.
def isNaN(x):
    return x != x  # True if NaN

def get_url(host, path, query):
    url = []
    for h, p, q in zip(host, path, query):
        # Force all to string safely
        h = str(h)
        p = str(p)
        q = str(q) if not isNaN(q) else ''
        
        if q == '':
            u = h + p
        else:
            u = h + p + '?' + q
        
        url.append(u)
    return url

host = waf_data['Host'].astype(str)
path = waf_data['URI']
query = waf_data['GET-Query']
9  url contains host,path,query
url_list = get_url(host, path, query)
10 # Convert to DataFrame
url = pd.DataFrame({'url': url_list})
url
11 feature engineerings from the columns
def NumericCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of numeric characters
    numeric = set("0123456789")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If numeric character is present
        # in set numeric
        if num in numeric:
            count = count + 1

    return count

def EnglishLetterCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of english letters
    engletter = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

    # Loop to traverse the numso can
    # in the given string
    for num in str:

        # If english letter is present
        # in set engletter
        if num in engletter:
            count = count + 1

    return count

def SpecialCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of special characters
    specialchar = set("!#$%&'()*+,-./:;<=>?@[\]^_`{|}~\"")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If special character is present
        # in set specialchar
        if num in specialchar:
            count = count + 1

    return count

def DotCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Dot
    dot = set(".")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If dot character is present
        # in set dot
        if num in dot:
            count = count + 1

    return count

def SemiColCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Semi-colon
    semicolon = set(";")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If semi-colon character is present
        # in set semicolon
        if num in semicolon:
            count = count + 1

    return count

def UnderscoreCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Underscore
    underscore = set("_")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If underscore character is present
        # in set underscore
        if num in underscore:
            count = count + 1

    return count

def QuesMarkCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Question Mark
    quesmark = set("?")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Question Mark character is present
        # in set QuesMark
        if num in quesmark:
            count = count + 1

    return count

def HashCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Hash Character
    hashchar = set("#")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Hash Character is present
        # in set hashchar
        if num in hashchar:
            count = count + 1

    return count

def EqualCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Equals to Character
    equalchar = set("=")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Equals to Character character is present
        # in set equalchar
        if num in equalchar:
            count = count + 1

    return count

def PercentCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Percentage Character
    percentchar = set("%")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Percentage Character is present
        # in set percentchar
        if num in percentchar:
            count = count + 1

    return count

def AmpersandCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Ampersand Character
    ampersandchar = set("&")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Ampersand Character is present
        # in set ampersandchar
        if num in ampersandchar:
            count = count + 1

    return count

def DashCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Dash Character
    dashchar = set("-")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Dash Character is present
        # in set dashchar
        if num in dashchar:
            count = count + 1

    return count

def DelimiterCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Delimiter Characters
    delim = set("(){}[]<>'\"")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Delimiter Character is present
        # in set delimiter
        if num in delim:
            count = count + 1

    str1 = str.lower()
    # In string, what is the count that <? occurs
    a = str1.count("<?")
    if a != 0:
        count = count-a

    str2 = str.lower()
    # In string, what is the count that ?> occurs
    b = str2.count("?>")
    if b != 0:
        count = count-b

    str3 = str.lower()
    # In string, what is the count that <% occurs
    c = str3.count("<%")
    if c != 0:
        count = count - c

    str4 = str.lower()
    # In string, what is the count that %> occurs
    d = str4.count("%>")
    if d != 0:
        count = count - d

    str5 = str.lower()
    # In string, what is the count that /* occurs
    e = str5.count("/*")

    str6 = str.lower()
    # In string, what is the count that */ occurs
    f = str6.count("*/")

    return count+a+b+c+d+e+f

def AtCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of At Character
    atchar = set("@")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If At Character is present
        # in set atchar
        if num in atchar:
            count = count + 1

    return count

def TildeCharCount(str):
    # Initializing count variable to 0
    count = 0

    # Creating a set of Tilde Character
    tildechar = set("~")

    # Loop to traverse the num
    # in the given string
    for num in str:

        # If Tilde Character character is present
        # in set tildechar
        if num in tildechar:
            count = count + 1

    return count

def DigitAlphabetRatio(str):

    digit = 0
    numeric = set("0123456789")

    for num in str:
        if num in numeric:
            digit = digit + 1

    alphabet = 0
    engletter = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    flag = -1
    for num in str:
        if num in engletter:
            alphabet = alphabet + 1

    if alphabet != 0:
        ratio = digit/alphabet
        return ratio

    else:
        return flag

def SpecialcharAlphabetRatio(str):
    schar = 0
    specialchar = set("!#$%&'()*+,-./:;<=>?@[\]^_`{|}~\"")

    for num in str:
        if num in specialchar:
            schar = schar + 1

    alphabet = 0
    engletter = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    flag = -1
    for num in str:
        if num in engletter:
            alphabet = alphabet + 1

    if alphabet != 0:
        ratio = schar / alphabet
        return ratio

    else:
        return flag

def UppercaseLowercaseRatio(str):
    ucase = 0
    uppercase = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    for num in str:
        if num in uppercase:
            ucase = ucase + 1

    lcase = 0
    lowercase = set("abcdefghijklmnopqrstuvwxyz")
    flag = -1

    for num in str:
        if num in lowercase:
            lcase = lcase + 1

    if lcase != 0:
        ratio = ucase / lcase
        return ratio

    else:
        return flag

def URLLength(str):
    length = len(str)
    #print ("The length of the URL is: ", length)
    return length

def Entropy(data, unit='natural'):
    base = {
        'shannon' : 2.,
        'natural' : math.exp(1),
        'hartley' : 10.
    }

    if len(data) <= 1:
        return 0

    counts = Counter()

    for d in data:
        counts[d] += 1

    ent = 0

    probs = [float(c) / len(data) for c in counts.values()]
    for p in probs:
        if p > 0.:
            ent -= p * math.log(p, base[unit])

    return ent

def CheckIPAsHostName(stri):
    parsed_url = urllib.parse.urlparse(stri)
    #print(parsed_url)
    h = parsed_url.netloc
    try:
        # Try to convert to IP address (IPv4 or IPv6)
        socket.inet_aton(h)  # For IPv4
        return True
    except (socket.error, TypeError):
        # If it fails, check for IPv6
        try:
            socket.inet_pton(socket.AF_INET6, h)
            return True
        except (socket.error, TypeError):
            return False

#Function to find the length of the host name Feature37

def HostNameLength(str):
    parsed_url = urllib.parse.urlparse(str)
    #print(parsed_url.netloc)
    return len(parsed_url.netloc)


#Function to find the length of the path of the URL Feature38

def PathLength(str):
    parsed_url = urllib.parse.urlparse(str)
    #print(parsed_url.path)
    return len(parsed_url.path)

def QueryLength(str):
    parsed_url = urllib.parse.urlparse(str)
    #print(parsed_url.query)
    return len(parsed_url.query)



#Function to find if there is https occurs in host name Feature36

def HttpsInHostName(str):
    parsed_url = urllib.parse.urlparse(str)
    hostname = parsed_url.netloc
    #print(hostname)
    hostname = hostname.lower()
    # In string, what is the count that // occurs
    count = 0
    count = hostname.count("https")
    if count == 0:
        #print("Not present")
        return False
    else:
        if count != 0:
            #print("Present")
            return True

def DomainURLRatio(str):
    urllength = len(str)

    parsed_url = urllib.parse.urlparse(str)
    domain = parsed_url.netloc
    domainlength = len(domain)
    flag = "Undefined"

    if urllength != 0:
        ratio = domainlength / urllength
        return ratio

    else:
        return flag

#Function to find TLD of the URL Feature30


def TLD(url):
    parsed = urllib.parse.urlparse(url)
    hostname = parsed.hostname or url
    
    if CheckIPAsHostName(hostname):
        return None  # or "IP address, no TLD"
    
    result = tldextract.extract(hostname)
    return result.suffix  # 'com', 'co.uk', etc.

def IsHashed(str):

    def is_hex(str):

        hex_digits = set(string.hexdigits)

        return all(c in hex_digits for c in str)

    ishash = False
    if len(str) ==16 or len(str) == 32 or len(str) == 64 and str.isdigit == True:
        ishash = True
        #print("Hashed")

    if (len(str) == 32 or len(str) == 64 or len(str) == 128) and is_hex(str) == True:
        ishash = True
        return True
    else:
        #print("Not Hashed")
        return False

def TLDInPath(url):
    try:
        parsed_url = urllib.parse.urlparse(url)
        path = parsed_url.path

        tld = get_tld(url, fix_protocol=True)
        return tld in path
    except TldDomainNotFound:
        # Couldn’t determine TLD, likely IP or malformed URL
        return False
    except Exception as e:
        # Optionally log or print the error
        # print(f"Error: {e}")
        return False


def TLDInSubdomain(url):
    try:
        res = get_tld(url, fix_protocol=True)
        subdomain = tldextract.extract(url).subdomain
        if res in subdomain:
            return True
        else:
            return False
    except TldDomainNotFound:
        # The URL does not contain a recognized TLD (e.g., IP address or malformed domain)
        return False
    except Exception as e:
        # Optionally log the error
        # print(f"Error: {e}")
        return False



def switch_all_letters(url):
    domains = []
    # url = get_tld(url, as_object=True, fix_protocol=True)

    domain = url
    a = []
    j = 0
    glyphs = homo_dictionary
    result1 = set()
    for ws in range(1, len(domain)):
        for i in range(0, (len(domain) - ws) + 1):
            win = domain[i:i + ws]
            j = 0
            while j < ws:
                c = win[j]
                if c in glyphs:
                    win_copy = win
                    for g in glyphs[c]:
                        win = win.replace(c, g)
                        result1.add(domain[:i] + win + domain[i + ws:])
                        win = win_copy
                j += 1

    result2 = set()
    for domain in result1:
        for ws in range(1, len(domain)):
            for i in range(0, (len(domain) - ws) + 1):
                win = domain[i:i + ws]
                j = 0
                while j < ws:
                    c = win[j]
                    if c in glyphs:
                        win_copy = win
                        for g in glyphs[c]:
                            win = win.replace(c, g)
                            result2.add(domain[:i] + win + domain[i + ws:])
                            win = win_copy
                    j += 1
    return list(result1 | result2)


#check Deep Feature Vowel_swap and called in Containment()

def vowel_swap(domain):
    vowels = 'aeiou'
    result = []

    for i in range(0, len(domain)):
        for vowel in vowels:
            if domain[i] in vowels:
                result.append(domain[:i] + vowel + domain[i+1:])

    return list(set(result))


#check Deep Feature bitsquatting and called in Containment()

def bitsquatting(domain):
    result = []
    masks = [1, 2, 4, 8, 16, 32, 64, 128]

    for i in range(0, len(domain)):
        c = domain[i]
        for j in range(0, len(masks)):
            b = chr(ord(c) ^ masks[j])
            o = ord(b)
            if (o >= 48 and o <= 57) or (o >= 97 and o <= 122) or o == 45:
                result.append(domain[:i] + b + domain[i+1:])

    return result


#check Deep Feature insertion and called in Containment()

def insertion(domain):
    result = []

    for i in range(1, len(domain)-1):
        for keys in keyboards:
            if domain[i] in keys:
                for c in keys[domain[i]]:
                    result.append(domain[:i] + c + domain[i] + domain[i+1:])
                    result.append(domain[:i] + domain[i] + c + domain[i+1:])

    return list(set(result))


#check Deep Feature omission and called in Containment()

def omission(domain):
    result = []

    for i in range(0, len(domain)):
        result.append(domain[:i] + domain[i+1:])

    return list(set(result))


#check Deep Feature repetition and called in Containment()

def repetition(domain):
    result = []

    for i in range(0, len(domain)):
        if domain[i].isalnum():
            result.append(domain[:i] + domain[i] + domain[i] + domain[i+1:])

    return list(set(result))


#check Deep Feature replacement and called in Containment()

def replacement(domain):
    result = []

    for i in range(0, len(domain)):
        for keys in keyboards:
            if domain[i] in keys:
                for c in keys[domain[i]]:
                    result.append(domain[:i] + c + domain[i+1:])

    return list(set(result))


#check Deep Feature subdomain and called in Containment()

def subdomain(domain):
    result = []

    for i in range(1, len(domain)-1):
        if domain[i] not in ['-', '.'] and domain[i-1] not in ['-', '.']:
            result.append(domain[:i] + '.' + domain[i:])

    return result


#check Deep Feature transpose and called in Containment()

def transposition(domain):
    result = []

    for i in range(0, len(domain)-1):
        if domain[i+1] != domain[i]:
            result.append(domain[:i] + domain[i+1] + domain[i] + domain[i+2:])

    return result


#check Deep Feature addition and called in Containment()

def addition(domain):
    result = []

    for i in range(97, 123):
        result.append(domain + chr(i))

    return result


#Function for the following features : Hyphenstring, Homoglyph, Vowel, Bitsquatting, Insertion, Omission, Repeatition, 
# Replacement, Subdomain, Transposition, Addition String : Feature52, Feature53, Feature54, Feature55,
#  Feature56, Feature57, Feature58, Feature59, Feature60, Feature61, Feature62

def Containment(str):
    val_without_tld = (str.rsplit('.', 1))[0]
    tld = (str.split('.'))[-1]
    #print(val_without_tld)


    hyphen_str = hyphenate_word(val_without_tld)

    if len(hyphen_str) == 0:
        hyphen_str = "No_hyphen"
    else:
        #print(hyphen_str)
        for domain in hyphen_str:
            #print(domain)
            if not is_registered(domain + '.' + tld):
                hyphen_str.remove(domain)
                if len(hyphen_str) != 0:
                    continue
                hyphen_str = "No_hyphen"

    #print(hyphen_str)



    homo_str = switch_all_letters(str)

    if len(homo_str) == 0:
        homo_str = "No_homo_str"
    else:
        for domain in homo_str:
            if not is_registered(domain + '.' + tld):
                homo_str.remove(domain)
                if len(homo_str) == 0:
                    homo_str = "No_homo_str"

    #print(homo_str)



    vowel_result = vowel_swap(val_without_tld)

    if len(vowel_result) == 0:
        vowel_result = "No_vowel_result"
    else:
        for domain in vowel_result:
            if not is_registered(domain + '.' + tld):
                vowel_result.remove(domain)
                if len(vowel_result) == 0:
                    vowel_result = "No_vowel_result"

    #print(vowel_result)



    bitsquatting_result = bitsquatting(val_without_tld)

    if len(bitsquatting_result) == 0:
        bitsquatting_result = "bitsquatting_result"
    else:
        for domain in bitsquatting_result:
            if not is_registered(domain + '.' + tld):
                bitsquatting_result.remove(domain)
                if len(bitsquatting_result) == 0:
                    bitsquatting_result = "bitsquatting_result"

    #print(bitsquatting_result)



    insertion_str = insertion(val_without_tld)

    if len(insertion_str) == 0:
        insertion_str = "No_insertion_str"
    else:
        for domain in insertion_str:
            if not is_registered(domain + '.' + tld):
                insertion_str.remove(domain)
                if len(insertion_str) == 0:
                    insertion_str = "No_insertion_str"

    #print(insertion_str)



    omission_str = omission(val_without_tld)

    if len(omission_str) == 0:
        omission_str = "No_omission_str"
    else:
        for domain in omission_str:
            if not is_registered(domain + '.' + tld):
                omission_str.remove(domain)
                if len(omission_str) == 0:
                    omission_str = "No_omission_str"

    #print(omission_str)



    repetition_str = repetition(val_without_tld)

    if len(repetition_str) == 0:
        repetition_str = "No_repetition_str"
    else:
        for domain in repetition_str:
            if not is_registered(domain + '.' + tld):
                repetition_str.remove(domain)
                if len(repetition_str) == 0:
                    repetition_str = "No_repetition_str"

    #print(repetition_str)



    replacement_str = replacement(val_without_tld)

    if len(replacement_str) == 0:
        replacement_str = "No_replacement_str"
    else:
        for domain in replacement_str:
            if not is_registered(domain + '.' + tld):
                replacement_str.remove(domain)
                if len(replacement_str) == 0:
                    replacement_str = "No_replacement_str"

    #print(replacement_str)



    subdomain_str = subdomain(val_without_tld)

    if len(subdomain_str) == 0:
        subdomain_str = "No_subdomain_str"
    else:
        for domain in subdomain_str:
            if not is_registered(domain + '.' + tld):
                subdomain_str.remove(domain)
                if len(subdomain_str) == 0:
                    subdomain_str = "No_subdomain_str"

    #print(subdomain_str)



    transposition_str = transposition(val_without_tld)

    if len(transposition_str) == 0:
        transposition_str = "No_transposition_str"
    else:
        for domain in transposition_str:
            if not is_registered(domain + '.' + tld):
                transposition_str.remove(domain)
                if len(transposition_str) == 0:
                    transposition_str = "No_transposition_str"

    #print(transposition_str)



    addition_str = addition(val_without_tld)

    if len(addition_str) == 0:
        addition_str = "No_addition_str"
    else:
        for domain in addition_str:
            if not is_registered(domain + '.' + tld):
                addition_str.remove(domain)
                if len(addition_str) == 0:
                    addition_str = "No_addition_str"

    #print(addition_str)
    arr = [hyphen_str, homo_str, vowel_result, bitsquatting_result, insertion_str, omission_str, repetition_str,
           replacement_str, subdomain_str, transposition_str, addition_str]
    return arr

# Your feature column names (based on the Containment output)
containment_columns = [
    'hyphen', 'homo_st', 'vowel_res', 'bitsquatting', 'insertion',
    'omission', 'repetition', 'replacement', 'subdomain',
    'transposition', 'addition'
]

# Calculate Levenshtein Distance

def LevenshteinDistanceDP(token1, token2):
    distances = numpy.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]






def Entropy(s):
    p, lns = Counter(s), float(len(s))

    return -sum( count/lns * math.log(count/lns, 2) for count in p.values())


#Function to check if www is used in url Feature04

def URLWithoutwww(url):
    url = url.lower()
    count = url.count("www.")
    if count == 0:
        flag = False
    else:
        flag = True
    return flag

def FTPUsed(self):
    self = self.lower()
    count = self.count("ftp://")
    if count == 0:
        flag = False
    else:
        flag = True
    return flag


#Function to check if files is used in url Feature07

def FilesInURL(self):
    self = self.lower()
    count = self.count("files")
    if count == 0:
        flag = False
    else:
        flag = True
    return flag

def JSUsed(self):
    self = self.lower()
    count = self.count(".js")
    if count == 0:
        flag = False
    else:
        flag = True
    return flag

#Function to check if .css is used in url Feature08

def CSSUsed(self):
    self = self.lower()
    count = self.count("css")
    if count == 0:
        flag = False
    else:
        flag = True
    return flag

def CheckEXE(str):
    res = distutils.spawn.find_executable(str)
    if res == None:
        return False
    else:
        return True

def FileExtension(self):
    count1 = self.count(".zip")
    count2 = self.count(".jpg")
    count3 = self.count(".gif")
    count4 = self.count(".rar")
    count5 = self.count("download.php")
    count6 = self.count("mail.php")
    count7 = self.count(".jar")
    count8 = self.count(".swf")
    count9 = self.count(".cgi")

    count = count1 + count2 + count3 + count4 + count5 + count6 + count7 + count8 + count9
    if count == 0:
        #print("No such word used")
        return False
    else:
        #print("Such file extension present")
        return True

def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return 1
    else:
        return 0

def suspicious_words(url):
    score_map = {
        'error': 30,
        'errorMsg': 30,
        'id': 10,
        'errorID': 30,
        'SELECT': 50,
        'FROM': 50,
        'WHERE': 50,
        'DELETE': 50,
        'USERS': 50,
        'DROP': 50,
        'CREATE': 50,
        'INJECTED': 50,
        'TABLE': 50,
        'alert': 30,
        'javascript': 20,
        'cookie': 25,
        '--': 30,
        '.exe': 30,
        '.php': 20,
        '.js': 10,
        'admin': 10,
        'administrator': 10,
        '\'': 30,
        'password': 15,
        'login': 15,
        'incorrect': 20,
        'pwd': 15,
        'tamper': 25,
        'vaciar': 20,
        'carrito': 25,
        'wait': 30,
        'delay': 35,
        'set': 20,
        'steal': 35,
        'hacker': 35,
        'proxy': 35,
        'location': 30,
        'document.cookie': 40,
        'document': 20,
        'set-cookie': 40,
        'create': 40,
        'cmd': 40,
        'dir': 30,
        'shell': 40,
        'reverse': 30,
        'bin': 20,
        'cookiesteal': 40,
        'LIKE': 30,
        'UNION': 35,
        'include': 30,
        'file': 20,
        'tmp': 25,
        'ssh': 40,
        'exec': 30,
        'cat': 25,
        'etc': 30,
        'fetch': 25,
        'eval': 30,
        'wait': 30,
        'malware': 45,
        'ransomware': 45,
        'phishing': 45,
        'exploit': 45,
        'virus': 45,
        'trojan': 45,
        'backdoor': 45,
        'spyware': 45,
        'rootkit': 45,
        'credential': 30,
        'inject': 30,
        'script': 25,
        'iframe': 25,
        'src=': 25,
        'onerror': 30,
        'prompt': 20,
        'confirm': 20,
        'eval': 25,
        'expression': 30,
        'function\(': 20,
        'xmlhttprequest': 30,
        'xhr': 20,
        'window.': 20,
        'document.': 20,
        'cookie': 25,
        'click': 15,
        'mouseover': 15,
        'onload': 20,
        'onunload': 20,
    }

    matches = re.findall(r'(?i)' + '|'.join(score_map.keys()), url)

    total_score = sum(score_map.get(match.lower(), 0) for match in matches)
    return total_score

9 balance_url = url
10 
balance_url['count_char_url'] = balance_url['url'].apply(NumericCharCount)
balance_url['en_let_url'] = balance_url['url'].apply(EnglishLetterCount)
balance_url['special_char_url'] = balance_url['url'].apply(SpecialCharCount)
balance_url['count_dot_url'] = balance_url['url'].apply(DotCount)
balance_url['semi_col_url'] = balance_url['url'].apply(SemiColCount)
balance_url['underscore_url'] = balance_url['url'].apply(UnderscoreCount)
balance_url['Ques_mark_url'] = balance_url['url'].apply(QuesMarkCount)
balance_url['Hash_char_url'] = balance_url['url'].apply(HashCharCount)
balance_url['Equal_url'] = balance_url['url'].apply(EqualCount)
balance_url['Parent_char_url'] = balance_url['url'].apply(PercentCharCount)
balance_url['Ampersand_url'] = balance_url['url'].apply(AmpersandCount)
balance_url['Dash_char_url'] = balance_url['url'].apply(DashCharCount)
balance_url['Delimiter_url'] = balance_url['url'].apply(DelimiterCount)
balance_url['At_char_url'] = balance_url['url'].apply(AtCharCount)
balance_url['Tilde_char_url'] = balance_url['url'].apply(TildeCharCount)
balance_url['DigAlp_ratio_url'] = balance_url['url'].apply(DigitAlphabetRatio)
balance_url['SpeAlp_ratio_url'] = balance_url['url'].apply(SpecialcharAlphabetRatio)
balance_url['UpLow_ratio_url'] = balance_url['url'].apply(UppercaseLowercaseRatio)
balance_url['IP_hostname_url'] = balance_url['url'].apply(CheckIPAsHostName)
balance_url['Host_len_url'] = balance_url['url'].apply(HostNameLength)
balance_url['Path_len_url'] = balance_url['url'].apply(PathLength)
balance_url['Query_len_url'] = balance_url['url'].apply(QueryLength)
# balance_url['Https_url'] = balance_url['url'].apply(HttpsInHostName)
balance_url['Domain_url_ratio'] = balance_url['url'].apply(DomainURLRatio)
balance_url['TLD_url'] = balance_url['url'].apply(TLD)
balance_url['Ishased_url'] = balance_url['url'].apply(IsHashed)
balance_url['Tld_path_url'] = balance_url['url'].apply(TLDInPath)
balance_url['Tld_subdomain_url'] = balance_url['url'].apply(TLDInSubdomain)
balance_url['Entropy_url'] = balance_url['url'].apply(Entropy)
balance_url['No_www_url'] = balance_url['url'].apply(URLWithoutwww)
balance_url['FTP_url'] = balance_url['url'].apply(FTPUsed)
balance_url['Files_in_url'] = balance_url['url'].apply(FilesInURL)
balance_url['JS_in_url'] = balance_url['url'].apply(JSUsed)
balance_url['CSS_in_url'] = balance_url['url'].apply(CSSUsed)
balance_url['Exe_check_url'] = balance_url['url'].apply(CheckEXE)
balance_url['File_ext_url'] = balance_url['url'].apply(FileExtension)
balance_url['shortened_url'] = balance_url['url'].apply(shortening_service)
balance_url['sus_words_url'] = balance_url['url'].apply(suspicious_words)
11 balance_url['Class'] = waf_data['Class']
balance_url
 12 # change object labeling to numbers
balance_url['Class'] = balance_url['Class'].map({'Valid': 0, 'Anomalous': 1})
13 balance_url
14 print(balance_url.dtypes)
15 print(balance_url.columns.tolist())
16 # balance_url['TLD_url'].isna().value_counts()
balance_url['TLD_url'].value_counts()
17 # TLD_url object type is changed to
# balance_url['TLD_url'] = balance_url['url'].apply(lambda x: tldextract.extract(x).suffix)
18 bool_cols = balance_url.select_dtypes(include='bool').columns
# Replace True with 0 and False with 1
balance_url[bool_cols] = balance_url[bool_cols].applymap(lambda x: 0 if x else 1)
@print(balance_url.dtypes)
19 lb_mke = LabelEncoder()
balance_url["TLD_url"] = lb_make.fit_transform(balance_url["TLD_url"])
20 balance_url['TLD_url'].value_counts()
21 x_url = balance_url.drop(columns=['Class','url'])
y_url = balance_url['Class']
22 y_url = y_url.fillna(0)   # or y_url = y_url.fillna('benign')
23 model = RandomForestClassifier(n_estimators=num_epoch, random_state=random_state)
model.fit(x_url, y_url)
24 # Confirm the model was trained
print("Model trained:", hasattr(model, "estimators_"))  # ✅ Should print True

# STEP 6: Get and plot feature importances
importances = model.feature_importances_
feature_names = x_url.columns
feature_names
25 importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(12, 7))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette='Spectral')
plt.title('Feature Importances ')
plt.tight_layout()
plt.show()
26 top_features = importance_df['Feature'].head(25).tolist()
X_top = balance_url[top_features]
X_top
27 waf_data['Class'] = waf_data['Class'].map({'Valid': 0, 'Anomalous': 1})
waf_data
28 waf_data["Method"].value_counts()
29 waf_data["Host-Header"].value_counts()
30 # for col in waf_data.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     waf_data[col] = le.fit_transform(waf_data[col])
lb_make = LabelEncoder()
waf_data["Method"] = lb_make.fit_transform(waf_data["Method"])
waf_data["Host-Header"] = lb_make.fit_transform(waf_data["Host-Header"])
31 sns.set_style('darkgrid')
sns.countplot(data=waf_data, x='Class')

32 # Total missing values per column
print("Missing Values Per Column:")
print(waf_data.isnull().sum())

# Percentage of missing values
print("\nPercentage of Missing Values:")
print((waf_data.isnull().mean() * 100).round(2))
33 waf_data = waf_data.fillna(0)
waf_data
34 waf_data.dtypes
35 X_top['Content-Length'] = waf_data['Content-Length']
X_top['Method'] = waf_data['Method']
X_top['Host-Header'] = waf_data['Host-Header']
X_top.columns
36 x_data = waf_data.drop(columns=['Class', 'Content-Length', 'Method', 'Host-Header'])
y_data = waf_data['Class']
#use Embedding Layers
37 def load_data(df, columns, max_len):
    """
    Converts multiple string columns in a DataFrame into ASCII-encoded
    and padded NumPy array suitable for neural network input.

    Args:
        df (DataFrame): Input DataFrame with text columns.
        columns (list): List of column names to embed.
        max_length (int): Max length per column after padding/truncating.

    Returns:
        np.ndarray: Combined ASCII-encoded array of shape (rows, columns * max_length)
    """
    all_encoded = []

    for col in columns:
        col_data = df[col].fillna('').astype(str).str.lower()
        col_encoded = []
        for item in col_data:
            decoded = unquote(item)  # URL-decode if needed
            ascii_vals = [ord(c) for c in decoded.strip()]
            ascii_vals = ascii_vals[:max_len]
            ascii_vals += [0] * (max_len - len(ascii_vals))  # padding
            col_encoded.append(ascii_vals)
        all_encoded.append(np.array(col_encoded))  # shape: (rows, max_length)

    # Concatenate all columns horizontally
    return np.concatenate(all_encoded, axis=1)  # shape: (rows, columns * max_length)
38 obj_features = list(x_data.columns)
X_ascii = load_data(x_data, obj_features, max_len)
X_ascii.shape
39 X_top.shape
40X_combined = np.concatenate([X_ascii, X_top], axis=1)
X_combined.shape
41 # Step 4: Train/Val/Test Split
x_train, x_temp, y_train, y_temp = train_test_split(X_combined, y_data, test_size=0.16, random_state=random_state)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=random_state)
42 print(f'x_train: {x_train.shape}, x_val: {x_val.shape}, x_test: {x_test.shape}')
43 # Step 5: Compute Class Weights
classes = np.unique(y_train)
classes
44 class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights_dict = dict(zip(classes, class_weights))
print("Class Weights:", class_weights_dict)
45 # Step 6: Callbacks including ModelCheckpoint
checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.8, verbose=1)
46 x_train = x_train.astype(int)
x_val = x_val.astype(int)
x_test = x_test.astype(int)
47 print("y_train:", y_train.shape, y_train.dtype)
print("y_val:", y_val.shape, y_val.dtype)
48 vocab_size = int(np.max(x_train)) + 1
vocab_size
49 # CNN MODEL
50 # Step 6: Build CNN_model 1
CNN_model1 = Sequential([Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
                         Conv1D(filters=64, kernel_size=3, activation='relu'),
                         GlobalMaxPooling1D(),
                         Dropout(0.5),
                         Dense(64, activation='relu'),
                         Dense(1, activation='sigmoid')])


# Step 6: Build CNN Model 2
CNN_model2 = Sequential([Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
                         Conv1D(filters=128, kernel_size=3, activation='relu'),  # Increased filters and kernel size
                         MaxPooling1D(pool_size=2),  # Added max pooling after the first conv layer
                         Conv1D(filters=64, kernel_size=3, activation='relu'),  # Second conv layer
                         GlobalMaxPooling1D(),  # Global pooling layer to reduce dimensionality
                         Dropout(0.5),  # Dropout for regularization
                         Dense(128, activation='relu'),  # Increased units in the dense layer
                         Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# # Step 6: Build CNN Model 2 with Dropout and Regularization
# CNN_model3 = Sequential([
#     Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
#     Conv1D(filters=128, kernel_size=5, activation='relu', kernel_regularizer=l2(0.01)),  # L2 regularization
#     MaxPooling1D(pool_size=2),  # Added max pooling layer
#     Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),  # L2 regularization
#     GlobalMaxPooling1D(),  # Global pooling to summarize features
#     Dropout(0.5),  # Dropout for regularization
#     Dense(128, activation='relu', kernel_regularizer=l2(0.01)),  # L2 regularization
#     Dropout(0.5),  # Additional dropout layer
#     Dense(1, activation='sigmoid')  # Output layer for binary classification
# ])


# Step 6: Build CNN Model 3
CNN_model3 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Conv1D(filters=256, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),  # Increased filters
    MaxPooling1D(pool_size=2),  # Max pooling layer
    Conv1D(filters=128, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),  # Second conv layer
    MaxPooling1D(pool_size=2),  # Additional pooling layer
    Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=l2(0.01)),  # Third conv layer
    GlobalMaxPooling1D(),  # Global pooling
    Dropout(0.5),  # Dropout for regularization
    Dense(64, activation='relu', kernel_regularizer=l2(0.01)),  # Dense layer with regularization
    Dropout(0.5),  # Additional dropout
    Dense(1, activation='sigmoid')  # Output layer
 
 51 LSTM MODEL
LSTM_model1 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])


# Step 6: Build LSTM Model 2
LSTM_model2 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128, return_sequences=True),   # LSTM with default 'tanh' activation
    Dropout(0.3),  # Slightly lower dropout
    LSTM(64, return_sequences=False),  # Second LSTM layer, with default 'tanh' activation
    Dropout(0.5),
    Dense(32, activation='relu'),  # Smaller dense layer
    Dense(1, activation='sigmoid')  # Output layer
])


# Step 6: Build Deeper LSTM Model 3
LSTM_model3 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128, return_sequences=True),  # First LSTM layer with 128 units
    Dropout(0.3),  # Dropout to prevent overfitting
    LSTM(64, return_sequences=True),  # Second LSTM layer with 64 units
    Dropout(0.3),  # Additional dropout
    LSTM(32, return_sequences=False),  # Third LSTM layer with 32 units
    Dropout(0.5),  # Dropout before dense layers
    Dense(32, activation='relu'),  # Dense layer with ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with Sigmoid activation for binary classification
])

52 HYBIRD CNN_LSTM MODEL
 cnn_lstm_hybrid = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, return_sequences=False),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_lstm_hybrid2 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    Conv1D(filters=128, kernel_size=5, activation='relu'),  # Increased filters and kernel size
    MaxPooling1D(pool_size=2),
    Dropout(0.3),  # Reduced dropout rate
    LSTM(128, return_sequences=True),  # LSTM with return_sequences=True
    Dropout(0.4),  # Adjusted dropout rate
    LSTM(64, return_sequences=False),  # Second LSTM layer
    Dense(32, activation='relu'),  # Smaller dense layer
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

53 # Step 6: Build LSTM Model 2
LSTM_model2 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128, return_sequences=True),   # LSTM with default 'tanh' activation
    Dropout(0.3),  # Slightly lower dropout
    LSTM(64, return_sequences=False),  # Second LSTM layer, with default 'tanh' activation
    Dropout(0.5),
    Dense(32, activation='relu'),  # Smaller dense layer
    Dense(1, activation='sigmoid')  # Output layer
])


# Step 6: Build Deeper LSTM Model 3
LSTM_model3 = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len),
    LSTM(128, return_sequences=True),  # First LSTM layer with 128 units
    Dropout(0.3),  # Dropout to prevent overfitting
    LSTM(64, return_sequences=True),  # Second LSTM layer with 64 units
    Dropout(0.3),  # Additional dropout
    LSTM(32, return_sequences=False),  # Third LSTM layer with 32 units
    Dropout(0.5),  # Dropout before dense layers
    Dense(32, activation='relu'),  # Dense layer with ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with Sigmoid activation for binary classification
])
54 # Step 7: Compile Model with Learning Rate
if selected_model == "CNN_model2":
    selected_model = CNN_model2
selected_model.compile(optimizer=Adam(learning_rate=lr_rate), loss='binary_crossentropy', metrics=['accuracy'])
#selected_model.summary()
55 # Step 9: Check if checkpoint exists and load weights
checkpoint_path = "checkpoint.weights.h5"
model_path = "model.h5"

# if os.path.exists(checkpoint_path):
#     os.remove(checkpoint_path)
# if os.path.exists(model_path):
#     os.remove(model_path)

# print("Old checkpoints removed.")

if os.path.exists(checkpoint_path):
    print("Loading saved model weights...")
    selected_model.load_weights(checkpoint_path)

56 # Step 10: Train Model with Class Weights and Callbacks
history = selected_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=num_epoch, batch_size=batch_size,
                             callbacks=[early_stop, reduce_lr, checkpoint], class_weight=class_weights_dict, verbose=1 )
57 # Step 11: Evaluate
test_loss, test_acc = selected_model.evaluate(x_test, y_test)
print(f"\n Test Accuracy: {test_acc:.4f}")
58 # Step 12: Predict & Metrics
y_pred_probs = selected_model.predict(x_test)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\n Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n Classification Report:")
print(classification_report(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
# plt.title("Confusion Matrix")
plt.title(f"{selected_model} Confusion Matrix")
plt.show()
