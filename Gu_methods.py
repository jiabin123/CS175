import json
from sklearn.feature_extraction.text import *
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC


def create_bow_from_reviews(filename):
    print('Loading the file:', filename)
    with open(filename, 'r') as jfile:
        data = json.load(jfile)
    print('Total number of reviews extracted =', len(data))

    text = []
    Y = []
    print('Extracting tokens from each review.....(can be slow for a large number of reviews)......')
    for d in data:  # can substitute data[0:9] here if you want to test this function on just a few reviews
        review = d['review']
        stars = int(d['rating'])
        if stars > 5:  # represent scores > 5 as "1" / positive
            score = 1
        else:          # represent scores > 5 as "0" / negative
            score = 0

        text.append(review)
        Y.append(score)

    # create an instance of a TF-IDFVectorizer, using
    # (1) the standard 'english' stopword set
    # (2) only keeping terms in the vocabulary that occur in at least 1% of documents
    # (3) allowing both unigrams and bigrams in the vocabulary (use "ngram_range=(1,2)" to do this)
    vectorizer = TfidfVectorizer()

    # create a sparse BOW array from 'text' using vectorizer
    X = vectorizer.fit_transform(text)

    print('Data shape: ', X.shape)

    # you can uncomment this next line if you want to see the full list of tokens in the vocabulary
    # print('Vocabulary: ', vectorizer.get_feature_names())

    return X, Y, vectorizer

def logistic_classification(X, Y, test_fraction):
    print('\nLogistic Classification:')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    #  set the state of the random number generator so that we get the same results across runs when testing our code

    # print('Number of training examples: ', X_train.shape[0])
    # print('Number of testing examples: ', X_test.shape[0])
    # print('Vocabulary size: ', X_train.shape[1])

    # Specify the logistic classifier model
    classifier = linear_model.LogisticRegression()

    # Train a logistic regression classifier and evaluate accuracy on the training data
    print('Training a model with', X_train.shape[0], 'examples.....')
    classifier.fit(X_train, Y_train)
    print('\nTraining:')
    train_accuracy = classifier.score(X_train, Y_train)
    print(' accuracy:', format(100 * train_accuracy, '.2f'))

    # Compute and print accuracy and AUC on the test data
    print('\nTesting: ')
    test_accuracy = classifier.score(X_test, Y_test)
    print(' accuracy:', format(100 * test_accuracy, '.2f'))

    class_probabilities = classifier.predict_proba(X_test)
    test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities[:, 1]);
    print(' AUC value:', format(100 * test_auc_score, '.2f'))

    return (classifier)

def support_vector_machine(X, Y, test_fraction):
    print('\nSupport Vector Machine:')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    #  set the state of the random number generator so that we get the same results across runs when testing our code

    # Specify the logistic classifier model
    classifier = SVC(probability=True)

    # Train a logistic regression classifier and evaluate accuracy on the training data
    print('Training a model with', X_train.shape[0], 'examples.....')
    classifier.fit(X_train, Y_train)
    print('\nTraining:')
    train_accuracy = classifier.score(X_train, Y_train)
    print(' accuracy:', format(100 * train_accuracy, '.2f'))

    # Compute and print accuracy and AUC on the test data
    print('\nTesting: ')
    test_accuracy = classifier.score(X_test, Y_test)
    print(' accuracy:', format(100 * test_accuracy, '.2f'))

    class_probabilities = classifier.predict_proba(X_test)
    test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities[:, 1]);
    print(' AUC value:', format(100 * test_auc_score, '.2f'))

    return (classifier)

def linear_support_vector_machine(X, Y, test_fraction):
    print('\nLinear Support Vector Machine:')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_fraction, random_state=42)
    #  set the state of the random number generator so that we get the same results across runs when testing our code

    # Specify the logistic classifier model
    classifier = LinearSVC()

    # Train a logistic regression classifier and evaluate accuracy on the training data
    print('Training a model with', X_train.shape[0], 'examples.....')
    classifier.fit(X_train, Y_train)
    print('\nTraining:')
    train_accuracy = classifier.score(X_train, Y_train)
    print(' accuracy:', format(100 * train_accuracy, '.2f'))

    # Compute and print accuracy and AUC on the test data
    print('\nTesting: ')
    test_accuracy = classifier.score(X_test, Y_test)
    print(' accuracy:', format(100 * test_accuracy, '.2f'))

    # class_probabilities = classifier.predict_proba(X_test)
    # test_auc_score = metrics.roc_auc_score(Y_test, class_probabilities[:, 1]);
    # print(' AUC value:', format(100 * test_auc_score, '.2f'))

    return (classifier)


def most_significant_terms(classifier, vectorizer, K):
    # find the largest K positive/negative weights' INDICES!
    coefs = classifier.coef_[0]
    topK_pos_indices = np.argsort(coefs)[-K:];
    topK_neg_indices = np.argsort(coefs)[0:K];

    topK_pos_weights = []
    topK_neg_weights = []
    topK_pos_terms = []
    topK_neg_terms = []

    # cycle through the indices, in the order of largest weight first
    # 1) append the weight and term to lists
    # 2) print K lines:
    #     (a) the term corresponding to the weight (a string)
    #     (b) the weight value itself (a scalar printed to 3 decimal places)
    print('Most significant positive terms & weight:')
    for i in topK_pos_indices[::-1]:
        weight = coefs[i]
        term = vectorizer.get_feature_names()[i]
        topK_pos_weights.append(weight)
        topK_pos_terms.append(term)
        print('term: {:<15}, weight = {:.4f}'.format(term, weight))

    print('Most significant negative terms & weight:')
    for i in topK_neg_indices:
        weight = coefs[i]
        term = vectorizer.get_feature_names()[i]
        topK_neg_weights.append(weight)
        topK_neg_terms.append(term)
        print('term: {:<15}, weight = {:.4f}'.format(term, weight))

    return (topK_pos_weights, topK_neg_weights, topK_pos_terms, topK_neg_terms)





X, Y, vectorizer_BOW = create_bow_from_reviews('reviews.json')
test_fraction = 0.5
logistic_classifier = logistic_classification(X, Y, test_fraction)
most_significant_terms(logistic_classifier, vectorizer_BOW, K=20)
SVC = support_vector_machine(X, Y, test_fraction)
    #most_significant_terms(SVC, vectorizer_BOW, K=20)
    # UNABLE to use coef_ attribute in the SVC
LSVC = linear_support_vector_machine(X, Y, test_fraction)
most_significant_terms(LSVC, vectorizer_BOW, K=20)





