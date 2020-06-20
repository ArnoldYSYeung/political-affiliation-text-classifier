# political_affiliation_text_classifier
 
## Summary

This project classifies text obtained from 23 Subreddits, categorized into 4 political affliations: Left, Center, Right, Alt.

This project is completed in 3 steps:
- <strong>Pre-Processing</strong> `preproc.py`: Text is pre-processed to remove stopwords and alter punctuations. Tokens are tagged based on their part-of-speech and lemmatization is applied using spaCy
- <strong>Feature Extraction</strong> `extract_features.py`: 30+ types of features are extracted from the pre-processed text based on characteristics of the text (e.g., numbers of different part-of-speech) and other text features such as pre-processed psychological LIWC features
- <strong>Classification</strong> `classify.py`: Text is split into training, validation, and test datasets. Different machine learning models (e.g., Support Vector Machine, Random Forest, Multi-Layer Perceptron, AdaBoost) are used to classify the text and results are compared (e.g., p-values)

This project is a modification of the original project for <a href="http://www.cs.toronto.edu/~frank/csc401/">CSC401/2511</a>. 

## Setup
The following dependencies are used:
- <a href="https://spacy.io/">spaCy 2.3.0</a>
- <a href="https://numpy.org/">NumPy 1.17</a>
- <a href="scikit-learn.org">Scikit-Learn 0.22.2</a>
- <a href="scipy.org">Scipy 1.13</a>

To run the code, the following data needs to be downloaded and stored in the appropriate directory:
- <a href="https://drive.google.com/drive/folders/1kiWXg8xyPoQik2goDtIDByh1JvsNj9C8?usp=sharing">Text data</a> in `./data/`
- <a href="https://drive.google.com/drive/folders/1QkxOINiFj-yqlgQp13TZiaEWUSiGqhzp?usp=sharing">Wordlists</a> in `./Wordlists/`
- <a href="https://drive.google.com/drive/folders/1kiWXg8xyPoQik2goDtIDByh1JvsNj9C8?usp=sharing">Pre-processed psychological features</a> in `./feats/`

## Run
`preproc.py` pre-processes the original text data and outputs a file containing the pre-processed text. `extract_features.py` takes the pre-processed text and extracts the relevant features for the text. `classify.py` deploys multiple machine learning models to classify the text and acts as the experiment pipeline.

## Documentation
Coming soon...
