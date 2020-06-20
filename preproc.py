"""
Name: Arnold Yeung
Date: February 11 2019
Description: Pre-processes Reddit comments from .json file using linguistic features 
"""


import sys
import argparse
import os
import json
import re
import string
import spacy
import html

indir = '/u/cs401/A1/data/';

spacy_model = 'en_core_web_sm'
indir = r'./data';

end_of_sentence_punc = ['.', '!', '?']

nlp = spacy.load(spacy_model, disable = ['parser', 'ner'])

def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment
    Steps 1 to 3: String pre-processing
    Steps 4 to 5: Tokenization

    Parameters:                                                                      
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step  

    Returns:
        modComm : string, the modified comment 
    '''

    # print(comment)

    modComm = ''

    if 1 in steps:              # remove newline characters s
        # print('Step 1: Removing new lines...')
        comment = comment.replace('\n', '')
        
    if 2 in steps:              #   convert html character codes to ASCII
        # print('Step 2: Converting HTML character codes to ASCII...')
        comment = html.unescape(comment)
        
    if 3 in steps:              # remove urls
        # print('Step 3: Removing URLs...')
        
        #   NOTE: Currently does not work on cases where 'http' or 'www' is preceded
        #           by a non-space character (e.g. "This link [http://www.youtube.com]")
        comment = ' '.join(token for token in comment.split(' ') if not 
                 token.startswith('http'))
        comment = ' '.join(token for token in comment.split(' ') if not 
                 token.startswith('www'))
        
#        comment = re.sub('\s^http\S+','', comment)
#        comment = re.sub('\s^www\S+','', comment)
    
    if 4 in steps:  #   split each punctuation into it's own token by adding whitespace
 
        # print('Step 4: Separating punctuations...')
        
        abbrev_file = './Wordlists/abbrev.english'
        
        #   get abbreviations
        abbreviations = []
        file = open(abbrev_file, 'r')
        for line in file.readlines():
            abbreviations.append(line[:-2]+'\.')         #  [:-1] removes '.\n' and add '\.' (. is a special character)
        file.close()
        abbreviations_regex = ''.join([r'(\s' + abbrev + r')|(^' 
                                       + abbrev + r')|' for abbrev in abbreviations])
        abbreviations_regex += '(\s(\w\.){2}(\w\.)*\w*)|(^(\w\.){2}(\w\.)*\w*)|'          #   add in 2+ letter abbreviations
        abbreviations_regex += '(\s[a-hj-zA-HJ-Z]\.)|(^[a-hj-zA-HJ-Z]\.)|'             #   add in single letter abbreviations (with exclusion of 'I.')
        abbreviations_regex += '(\s[0-9]+\.[0-9]+)|(^[0-9]+\.[0-9]+)|'
        
        #   get punctuations
        punctuations = string.punctuation.replace("'", "")
        punctuations_regex = '([' + punctuations + ']+)'  # sequence of/single punctuations 
        
        regex = r'('+abbreviations_regex + punctuations_regex+r')'
        comment = re.sub(regex, r' \1 ', comment)
    
    if 5 in steps:      #   split clitics using whitespace
        
        #   print('Step 5: Splitting clitics...')
        
        clitic_file = os.path.abspath(os.path.join(__file__ ,"../../.."))+r'\Wordlists\clitics'
        
        #   NOTE: A look-up table might be more accurate for common clitics
        
        #   The below code draws from a provided list of clitics:
#        clitic_endings = []
#        file = open(clitic_file, 'r')
#        for line in file.readlines():
#            clitic_endings.append(line.replace('\n', ''))
#        file.close()
#        front_apostrophe = [clitic for clitic in clitic_endings if clitic[0] == "'"]
#        end_apostrophe = [clitic for clitic in clitic_endings if clitic[-1] == "'"]
#        between_apostrophe = ["n't"]
#        clitic_list = [front_apostrophe, between_apostrophe, end_apostrophe]
        
#        clitic_regex = ''.join([r'(' + clitic + r')|' 
#                                for clitic in clitic_endings if clitic[0]=="'"])
        
        #   special case: n't
        regex = r"(\s[a-zA-z]+)(n't)"
        comment = re.sub(regex, r" \1 \2 ", comment)
        regex = r"(^[a-zA-z]+)(n't)"
        comment = re.sub(regex, r" \1 \2 ", comment)
        #   We assume that all words with ' in between are clitics (this covers newspeak)
        # for clitics that do not end with 'n' in first half
        regex = r"(\s[a-zA-Z]*[a-mo-zA-Z]+)\'([a-zA-Z]+)"
        comment = re.sub(regex, r" \1 '\2 ", comment)
        regex = r"(^[a-zA-Z]*[a-mo-zA-Z]+)\'([a-zA-Z]+)"
        comment = re.sub(regex, r" \1 '\2 ", comment)
        # for clitics that do end with 'n' in first half, but have more than 2 letters
        regex = r"(\s[a-zA-Z][a-zA-Z]+)\'([a-zA-Z]+)"
        comment = comment = re.sub(regex, r" \1 '\2 ", comment)
        regex = r"(^[a-zA-Z][a-zA-Z]+)\'([a-zA-Z]+)"
        comment = comment = re.sub(regex, r" \1 '\2 ", comment)
    
    #   tokenize using whitespace
    #   NOTE: This step is kept outside if-statements so that tokenization 
    #           occurs regardless of whether steps 4 or 5 are selected
    tokens = comment.strip().split()
    
    if 6 in steps:  #   tag part-of-speech for each token
        
        # print('Step 6: Tagging parts-of-speech...')
        
        #   Load English pipeline and disable parser and name-entity recognition
        # nlp = spacy.load(spacy_model, disable = ['parser', 'ner'])
        doc = spacy.tokens.Doc(nlp.vocab, tokens)
        doc = nlp.tagger(doc)
        
        tokens = []
        for token in doc:
            tokens.append(token.text+'/'+token.tag_)
    
    if 7 in steps:  #   remove stop words
        
        # print('Step 7: Removing stop words...')
        
        stopwords_file = './Wordlists/StopWords'
        
        #   load stop words
        stop_words = []
        file = open(stopwords_file, 'r')
        for line in file.readlines():
            stop_words.append(line[:-1])         #  [:-1] removes '.\n' and add '\.' (. is a special character)
        file.close()
        
        idx_to_remove = []
        for i in range(0, len(tokens)):
            #   get word up to '/'
            word = tokens[i].strip().split('/') 
            if word[0].lower() in stop_words:       #   changes word to all lowercase to avoid capitalized words (e.g. 'The') from staying
                idx_to_remove.append(i)
        remove_idx_from_list(tokens, idx_to_remove)

    if 8 in steps:  #   lemmatization 
        
        #   NOTE: Lemmatization data from step 6 is not used to avoid dependence
        #   (e.g. not running step 6) and change of token indices due to removal
        #   of stopwords
        
        # print('Step 8: Lemmatizing...')
        
        #   extract word before '/'
        words = []
        tags = []
        
        for word in tokens:
            word_segment = word.strip().split('/')
            if len(word_segment) >= 2:              #   if there are tags
                words.append(word_segment[0])
                tags.append(word_segment[1])
            else:
                words.append(word_segment[0])
        #   Note that the above method would remove and split all '/' which belong
        #   to actual words.  This is a flaw in the design.
        words = [word for word in words if word != '']  #   remove all empty cells which may be caused

        #   lemmatize words
        # nlp = spacy.load(spacy_model, disable = ['parser', 'ner'])
        doc = spacy.tokens.Doc(nlp.vocab, words)
        doc = nlp.tagger(doc)
        
        #   change words to lemmatized version
        tokens = []
        for i in range(0, len(doc)):
            word = doc[i].lemma_
            if words[i][0].isupper():   #   if the word is capitalized
                #   capitalize the first letter of the lemma
                word = word.capitalize()
            if not tags:                    #   if there are no tags
                tokens.append(word)
            else:
                tokens.append(word + '/' + tags[i])
        
    if 9 in steps:          #   add new line between each sentence
        #   Rules for defining sentence:
        #       1. Sentences end with either [., !, ?, ...]
        #       2. Sentence-ending punctuations may be followed by other punctuations
        #           e.g. ?!!!?!, ."
        #       3. Abbreviations are less like end of sentences.
        #           (Assuming abbreviations is like any other word, probability of an
        #           abbreviation being the end of a sentence is less than probability of
        #           being in the middle of a sentence.)
        #       4. Periods (not part of abbreviations), exclamation marks, and question 
        #           marks (regardless of what following upper/lowercase) are most 
        #           likely end of sentences. (not all cases though)
        #       5. '...' followed by lowercase is NOT end of sentence.
        
        #
        #   Strategy: Add ' \n' to the end of tokens which denote end-of-sentences
        # print('Step 9: Separating sentences by line...')
        
        for i in range(0, len(tokens)-1):
            curr_token = tokens[i]
            next_token = tokens[i+1]          
            
            regex = r"\.\.+"
            word_segments = curr_token.strip().split('/')
            if len(word_segment) >= 2:              #   if there are tags
                word = word_segments[0]
            else:
                word = word_segments
            #   check if curr_token == '...' and is end of sentence
            if bool(re.match(regex, word)) is True:
                if next_token[0].isupper():
                    tokens[i] = curr_token + ' \n'
            elif curr_token[0] in end_of_sentence_punc: # for all other EOS punctuations
                tokens[i] = curr_token + ' \n'
        
    if 10 in steps:     #   convert text to lowercase
        # print('Step 10: Converting to lowercase...')
        for i in range(0, len(tokens)):
            word_segments = tokens[i].split('/')
            if len(word_segment) >= 2:              #   if there are tags
                word = word_segments[0].lower()
                tag = '/' + word_segments[1]
            else:
                word = word_segments.lower()
                tag = ''
            tokens[i] = word + tag
 

    modComm = " ".join(tokens)
       
    return modComm

def remove_idx_from_list(lst, idx_list):    
    num_removed = 0    
    for idx in idx_list:
        idx = idx-num_removed           #   account for new list length due to
                                        #       indices already removed
        assert idx < len(lst)           #   make sure index exist in list
        lst.pop(idx)
        num_removed += 1
    
def remove_keys(dictionary, keys):
    
    for key in keys:
        if key in dictionary:
            dictionary.pop(key)


def main( args ):

    allOutput = []
    arguments = args
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))
            #   Select appropriate args['max'] lines
            print('There are ' + str(len(data)) + ' comments.')
            start_idx = 0
            end_idx = start_idx + arguments['max']
            
            keys_to_remove = ['subreddit', 'ups', 'distinguished', 'parent_id', 'link_id',
                'retrieved_on', 'score_hidden', 'score', 'author_flair_css_class', 
                'archived', 'author_flair_text', 'created_utc', 'downs', 'subreddit_id',
                'gilded', 'edited', 'name', 'removal_reason', 'controversiality']
            
            print('Start Idx: ' + str(start_idx) + ', End Idx: ' + str(end_idx))
            
            for i in range(start_idx, end_idx):
                # read those lines with something like `j = json.loads(line)`
                
                #   output progress
                if i % 100 == 0: 
                    print(i, end=' ')
                
                linestring = data[i]                #   type: str
                line = json.loads(linestring)       #   convert from str to dict        
                
                # TODO: choose to retain fields from those lines that are relevant to you
                remove_keys(line, keys_to_remove)
                # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
                line['cat'] = file                # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
                mod_body = preproc1(line['body'])
                # TODO: replace the 'body' field with the processed text
                line['body'] = mod_body
                # print(line)
                # TODO: append the result to 'allOutput'
                allOutput.append(line)
    fout = open(args['output'], 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    """
    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()
    """
    
    args = {'max':          10000,
            'output':       'preproc.json'
            }
    
    if (args['max'] > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
