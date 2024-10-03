import json
import collections
import argparse
import random
import numpy as np

from util import *

random.seed(42)

def extract_unigram_features(ex):
    """Return unigrams in the hypothesis and the premise.
    Parameters:
        ex : dict
            Keys are gold_label (int, optional), sentence1 (list), and sentence2 (list)
    Returns:
        A dictionary of BoW featurs of x.
    Example:
        "I love it", "I hate it" --> {"I":2, "it":2, "hate":1, "love":1}
    """
    # BEGIN_YOUR_CODE
    # combine sentences
    combo_sent = ex['sentence1'] + ex['sentence2']

    # count words
    bow = collections.Counter(combo_sent)
    
    return dict(bow)
    # END_YOUR_CODE

def extract_custom_features(ex):
    """Design your own features.
    """
    # BEGIN_YOUR_CODE
    # initialize feature dict (bow in unigram)
    features = {}
    
    sentence1 = ex['sentence1']
    sentence2 = ex['sentence2']
    
    # unigram with all lowercase
    all_words = sentence1 + sentence2
    for word in set(all_words):
        features[f'unigram_{word.lower()}'] = all_words.count(word)
    
    # bigram
    def get_bigrams(sentence):
        return [f"{sentence[i]}_{sentence[i+1]}" for i in range(len(sentence)-1)]
    
    bigrams1 = get_bigrams(sentence1)
    bigrams2 = get_bigrams(sentence2)
    all_bigrams = bigrams1 + bigrams2
    for bigram in set(all_bigrams):
        features[f'bigram_{bigram.lower()}'] = all_bigrams.count(bigram)
    
    # negation
    negation_words = set(['not', 'no', 'never', 'neither', 'nor', 'none'])
    features['negation_s1'] = sum(1 for word in sentence1 if word.lower() in negation_words)
    features['negation_s2'] = sum(1 for word in sentence2 if word.lower() in negation_words)
    
    return features
    # END_YOUR_CODE

def learn_predictor(train_data, valid_data, feature_extractor, learning_rate, num_epochs):
    """Running SGD on training examples using the logistic loss.
    You may want to evaluate the error on training and dev example after each epoch.
    Take a look at the functions predict and evaluate_predictor in util.py,
    which will be useful for your implementation.
    Parameters:
        train_data : [{gold_label: {0,1}, sentence1: [str], sentence2: [str]}]
        valid_data : same as train_data
        feature_extractor : function
            data (dict) --> feature vector (dict)
        learning_rate : float
        num_epochs : int
    Returns:
        weights : dict
            feature name (str) : weight (float)
    """
    # BEGIN_YOUR_CODE
    # initialize weight dict
    weights = {}
    
    for epoch in range(num_epochs): 
        for ex in train_data:
            # get BoW for each ex
            bow = feature_extractor(ex)
            
            # get label
            y = ex['gold_label']
            
            # predict prob with current weights
            y_pred = predict(weights, bow)
            
            # update weights
            for f, v in bow.items():
                # calc gradient
                gradient = (y_pred - y) * v #f_w(x^{(i)})-y^{(i)}*ϕ(x^{(i)})
                # update
                weights[f] = weights.get(f, 0) - learning_rate * gradient
            
        # calc training & validation error
        train_pred = lambda ex: 1 if predict(weights, feature_extractor(ex)) > 0.5 else 0
        train_error = evaluate_predictor([(ex, ex['gold_label']) for ex in train_data], train_pred)
        valid_error = evaluate_predictor([(ex, ex['gold_label']) for ex in valid_data], train_pred)
        
        # display
        print(f'Epoch {epoch + 1}: train error = {train_error:.3f}, valid error = {valid_error:.3f}')
            
    return weights
    # END_YOUR_CODE

def count_cooccur_matrix(tokens, window_size=4):
    """Compute the co-occurrence matrix given a sequence of tokens.
    For each word, n words before and n words after it are its co-occurring neighbors.
    For example, given the tokens "in for a penny , in for a pound",
    the neighbors of "penny" given a window size of 2 are "for", "a", ",", "in".
    Parameters:
        tokens : [str]
        window_size : int
    Returns:
        word2ind : dict
            word (str) : index (int)
        co_mat : np.array
            co_mat[i][j] should contain the co-occurrence counts of the words indexed by i and j according to the dictionary word2ind.
    """
    # BEGIN_YOUR_CODE
    # dict for words & corresponding indices (w=word, i=index)
    word2ind = {w: i for i, w in enumerate(set(tokens))}
    
    # initialize co-occurence matrix
    vocab_len = len(word2ind)
    co_mat = np.zeros((vocab_len, vocab_len), dtype = int)
    
    for i, t in enumerate(tokens):
        token_ind = word2ind[t] #gets index of current token
        
        # window range
        start = max(0, i - window_size)
        end = min(len(tokens), i + window_size + 1)
        
        # count co-occurrences
        for n in range(start, end):
            if i != n: #skip current token
                neighbor = tokens[n]
                neighbor_ind = word2ind[neighbor]
                co_mat[token_ind][neighbor_ind] += 1
                
    return word2ind, co_mat
    # END_YOUR_CODE

def cooccur_to_embedding(co_mat, embed_size=50):
    """Convert the co-occurrence matrix to word embedding using truncated SVD. Use the np.linalg.svd function.
    Parameters:
        co_mat : np.array
            vocab size x vocab size
        embed_size : int
    Returns:
        embeddings : np.array
            vocab_size x embed_size
    """
    # BEGIN_YOUR_CODE
    # truncated SVD - truncate based on embed_size
    U, s, Vt = np.linalg.svd(co_mat, full_matrices=False)
    
    # truncate to the specified embedding size
    U_truncated = U[:, :embed_size]
    s_truncated = s[:embed_size]
     
    # compute the embeddings
    embeddings = U_truncated * s_truncated[np.newaxis, :]
    
    # sign
    signs = np.sign(embeddings[0, :])
    embeddings *= signs[np.newaxis, :]
    
    # sign second col
    embeddings[:, 1] *= -1
        
    return embeddings
    # END_YOUR_CODE

def top_k_similar(word_ind, embeddings, word2ind, k=10, metric='dot'):
    """Return the top k most similar words to the given word (excluding itself).
    You will implement two similarity functions.
    If metric='dot', use the dot product.
    If metric='cosine', use the cosine similarity.
    Parameters:
        word_ind : int
            index of the word (for which we will find the similar words)
        embeddings : np.array
            vocab_size x embed_size
        word2ind : dict
        k : int
            number of words to return (excluding self)
        metric : 'dot' or 'cosine'
    Returns:
        topk_words : [str]
    """
    # BEGIN_YOUR_CODE
    # get embedding of target word
    target_embed = embeddings[word_ind]
    
    # compute similarities
    if metric == 'dot':
        similarities = np.dot(embeddings, target_embed)
    elif metric == 'cosine':
        norm_target = np.linalg.norm(target_embed)
        norm_embeds = np.linalg.norm(embeddings, axis=1)
        similarities = np.dot(embeddings, target_embed) / (norm_embeds * norm_target)
    else: #error if neither dot or cosine
        raise ValueError("Metric must be either 'dot' or 'cosine'")
    
    # adjust similarity btwn word & itself to be very low
    similarities[word_ind] = float('-inf')
    
    # get indices of top k similar words
    top_k_inds = np.argsort(similarities)[-k:][::-1]
    
    # get words associated with indices (w=word, i=index)
    ind2word = {i: w for w, i in word2ind.items()}
    topk_words = [ind2word[i] for i in top_k_inds]
    
    return topk_words
    # END_YOUR_CODE
