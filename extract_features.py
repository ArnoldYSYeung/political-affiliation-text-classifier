"""
Name: Arnold Yeung
Date: February 11 2019
Description: Extracts relevant linguistic features from pre-processed comments
"""

import numpy as np
import sys
import argparse
import os
import json
import csv
import re
import string
import datetime


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''
    
    #   create feature value array
    num_features = 173
    features = np.ones((num_features, ), dtype=float)*-1    #   features with this value after extraction have error
    
    #   extract words from Wordlists features (see wordlist_files for index of features)
    wordlist_dir = './Wordlists'
    wordlist_files = ['First-person', 'Second-person', 'Third-person', 'Slang2']
    
    wordlist_features = []
    for i in range(0, len(wordlist_files)):
        wordlist_feature_set = []
        file = open(wordlist_dir + '/' + wordlist_files[i], 'r')
        for line in file.readlines():
            if len(line) > 1:    
                wordlist_feature_set.append(line[:-1])         #  [:-1] removes '\n'
        file.close()
        wordlist_features.append(wordlist_feature_set)
    
    #   create token lists (words and POS tags)
    tokens = comment.strip().split()
    words = []
    pos_tags = []
    for token in tokens:
        word, pos_tag = token.strip().split('/')
        words.append(word)
        pos_tags.append('/' + pos_tag)
    
    #   F1: number of first-person pronouns
    features[0] = count_matching_lists(words, wordlist_features[0])
    
    #   F2: number of second-person pronouns
    features[1] = count_matching_lists(words, wordlist_features[1])
    
    #   F3: number of third-person pronouns
    features[2] = count_matching_lists(words, wordlist_features[2])
    
    #   F4: number of coordinating conjunctions
    features[3] = pos_tags.count('/CC')
    
    #   F5: number of past-tense verbs
    features[4] = pos_tags.count('/VBD')
    
    #   F6: number of future-tense verbs ('ll, will, gonna, going+to+VB)
    features[5] = count_matching_lists(words, ["'ll", "will", "gonna"])
    #   special case for 'going+to+VB'
    for idx in find_idx_sequence_in_list(words, ['going', 'to']):
        if len(pos_tags) > idx+2:
            if pos_tags[idx+2] == '/VB':
                features[5] += 1
    
    #   F7: number of commas
    features[6] = pos_tags.count('/,')
    
    #   F8: number of multi-character punctuation tokens
    
    punctuations = string.punctuation.replace("'", "")
    regex = '([' + punctuations + '][' + punctuations + ']+)'  # sequence of multi-punctuations 
    features[7]  = len(re.findall(regex, " ".join(words)))
    
    #   F9: number of common nouns
    features[8] = count_matching_lists(pos_tags, ['/NN', '/NNS'])
    
    #   F10: number of proper nouns
    features[9] = count_matching_lists(pos_tags, ['/NNP', '/NNPS'])
    
    #   F11: number of adverbs
    features[10] = count_matching_lists(pos_tags, ['/RB', '/RBR', '/RBS'])
    
    #   F12: number of wh- words
    features[11] = count_matching_lists(pos_tags, ['/WDT', '/WP', '/WP$', 'WRB'])
    
    #   F13: number of slang acronyms
    features[12] = count_matching_lists(words, wordlist_features[3]) 

    #   F14: number of words in uppercase
    features[13] = len([word for word in words if word.isupper() and 
            len(word) >= 3])

    #   F15: average length of sentences, in tokens
    sentences = comment.strip().split('\n')
    sum = 0
    for sentence in sentences:
        sum += len(sentence.strip().split())
    features[14] = float(sum)/float(len(sentences))

    #   F16: average length of tokens, excluding punctation-only tokens,
    #           in characters
    regex = '([' + punctuations + ']+)'  # sequence of single/multi punctuations 
    punc_tokens = re.findall(regex, " ".join(words))    # get all punctuations in list of words
    
    #   remove punctuations from list of words
    non_punc_words = words
    for punctuation in punc_tokens:
        non_punc_words = [word for word in non_punc_words if word != punctuation] 
    
    sum = 0
    for word in non_punc_words:
        word = re.sub('[' + punctuations + ']', '', word)   #   remove all punctuations from words
        sum += len(word)
    if len(non_punc_words) == 0:
        features[15] = 0
    else:
        features[15] = float(sum)/len(non_punc_words)
    
    #   F17: number of sentences
    features[16] = len(sentences)
    
    #   get AoA, IMG, FAM from Bristol, Gilhooly, and Logie norms
    bgl_file = './Wordlists/BristolNorms+GilhoolyLogie.csv'
    bgl_dict = {}
    with open(bgl_file, newline = '') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)                    #   skips header
        for row in csv_reader:
            if row[1] != '':
                bgl_dict[row[1]] = list(map(int, row[3:6]))
    
    #   F18-20: average for Bristol, Gilhooly, and Logie norms
    AoA_sum = 0
    IMG_sum = 0
    FAM_sum = 0
    count = 0
    for word in words:
        if word in bgl_dict:
            AoA_sum += bgl_dict[word][0]
            IMG_sum += bgl_dict[word][1]
            FAM_sum += bgl_dict[word][2]
            count += 1
    if count == 0:          #   prevent division by 0
        count += 1   
    AoA_mean = float(AoA_sum) / count
    IMG_mean = float(IMG_sum) / count
    FAM_mean = float(FAM_sum) / count
    features[17] = AoA_mean
    features[18] = IMG_mean
    features[19] = FAM_mean
    
    #   F21-23: standard deviation for Bristol, Gilhooly, and Logie norms
    AoA_sum = 0
    IMG_sum = 0
    FAM_sum = 0
    count = 0
    for word in words:
        if word in bgl_dict:
            AoA_sum += float(bgl_dict[word][0] - AoA_mean)**2
            IMG_sum += float(bgl_dict[word][1] - IMG_mean)**2
            FAM_sum += float(bgl_dict[word][2] - FAM_mean)**2
            count += 1
    if count == 0:          #   prevent division by 0
        count += 1   
    features[20] = (AoA_sum / count)**(0.5)
    features[21] = (IMG_sum / count)**(0.5)
    features[22] = (FAM_sum / count)**(0.5)
    
    #   get Valence, Arousal, Dominance from Warringer norms
    warr_file = './Wordlists/Ratings_Warriner_et_al.csv'
    warr_dict = {}
    with open(warr_file, newline = '') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)                    #   skips header
        for row in csv_reader:
            if row[1] != '':
                warr_dict[row[1]] = []
                warr_dict[row[1]].append(float(row[2]))
                warr_dict[row[1]].append(float(row[5]))
                warr_dict[row[1]].append(float(row[8]))
    
    #   F24-26: average for Warringer norms
    V_sum = 0
    A_sum = 0
    D_sum = 0
    count = 0
    for word in words:
        if word in warr_dict:
            V_sum += warr_dict[word][0]
            A_sum += warr_dict[word][1]
            D_sum += warr_dict[word][2]
            count += 1
    if count == 0:          #   prevent division by 0
        count += 1      
    V_mean = float(V_sum) / count
    A_mean = float(A_sum) / count
    D_mean = float(D_sum) / count
    features[23] = V_mean
    features[24] = A_mean
    features[25] = D_mean
    
    #   F26-29: standard deviation for Warringer norms
    V_sum = 0
    A_sum = 0
    D_sum = 0
    count = 0
    for word in words:
        if word in warr_dict:
            V_sum += float(warr_dict[word][0] - V_mean)**2
            A_sum += float(warr_dict[word][1] - A_mean)**2
            D_sum += float(warr_dict[word][2] - D_mean)**2
            count += 1
    if count == 0:          #   prevent division by 0
        count += 1   
    features[26] = (V_sum / count)**(0.5)
    features[27] = (A_sum / count)**(0.5)
    features[28] = (D_sum / count)**(0.5)
    
    return features

def find_idx_sequence_in_list(in_list, sequence):
#   Returns index of start of sequence for every sequence in list
#   Note that this function has not been tested for repeating values
    
    #   get the indices of the sequence[0]
    indices = [idx for idx, value in enumerate(in_list) if value == sequence[0]]
    
    for i in range(1, len(sequence)):    #   for sequence[i]
        new_indices = []
        for idx in indices:
            if len(in_list) > idx+i:    #   make sure that index exist
                if in_list[idx+i] == sequence[i]:
                    new_indices.append(idx)
        indices = new_indices
    return indices   
    

def count_matching_lists(in_list, idx_list):
#   Count the number of occurrences within a list of elements in another list
    count = 0
    for element in idx_list:
        count += in_list.count(element)
    return count

def main( args ):

    data = json.load(open(args['in']))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here
    
    print("Start time: " + str(datetime.datetime.now()))
    
    LIWC_dir = './feats'
    
    #   load LIWC files
    print("Loading LIWC...")
    
    LIWC_files = {'Alt': ['Alt_IDs.txt', 'Alt_feats.dat.npy'],
                  'Left': ['Left_IDs.txt', 'Left_feats.dat.npy'],
                  'Center': ['Center_IDs.txt', 'Center_feats.dat.npy'],
                  'Right': ['Right_IDs.txt', 'Right_feats.dat.npy']}
    
    LIWC_feat_arr = {}
    LIWC_id_arr = {}
    
    for key in LIWC_files:
        
        LIWC_id_arr[key] = []        
        file = open(LIWC_dir + '/' + LIWC_files[key][0] , 'r')
        for line in file.readlines():
            LIWC_id_arr[key].append(line[:-1])         #  [:-1] removes '.\n' and add '\.' (. is a special character)
        file.close()        
        LIWC_feat_arr[key] = np.load(LIWC_dir + "/" + LIWC_files[key][1])         

    print('There are ' + str(len(data)) + ' comments.')    
    for i in range(0, len(data)):       #   for each comment
        
        #   output progress
        if i % 100 == 0: 
            print(i, end=' ')
        
        line = data[i]                #   type: dict
        text_features = np.transpose(np.asarray(extract1(line['body'])[0:29]))      #   convert features 1 to 29 to  np.array
        index = LIWC_id_arr[line['cat']].index(line['id'])
        LIWC_features = np.transpose(LIWC_feat_arr[key][index])
        
        #   add category to last column
        if line['cat'] == 'Left':
            category = int(0)
        elif line['cat'] == 'Center':
            category = int(1)
        elif line['cat'] == 'Right':
            category = int(2)
        else:
            category = int(3)
            
        LIWC_features = np.append(LIWC_features, [category])
        
        feats[i] = np.hstack((text_features, LIWC_features))
    
    print("End time: " + str(datetime.datetime.now()))
    
    #   delete because it takes up too much memory
    del LIWC_feat_arr
    del LIWC_id_arr
    
    print("Saving file...")
    np.savez_compressed( args['out'], feats)

    
if __name__ == "__main__": 

    """
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
    """                 

    args = {'in':       'preproc.json',
            'out':      'out.json'}

    main(args)

