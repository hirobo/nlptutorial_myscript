# coding: utf-8
from collections import defaultdict
import math
import ch01_test_unigram

def load_model(model_file):
    # format is same to the unigram model
    return ch01_test_unigram.load_model(model_file)

def test_bigram(model_file, test_file):
    H = 0 # entropy
    W = 0 # word count

    lambda_1 = 0.95 # I don't know which value is the best
    lambda_2 = 0.95 # I don't know which value is the best

    probs = load_model(model_file)
    V = len(probs) # total words

    for line in open(test_file, "r"):
        words = line.strip().split()
        words.insert(0, "<s>")
        words.append("</s>")
        for i in range(1, len(words)): # beggins after <s>
            w_i1 = words[i-1]
            w_i = words[i]
            n_gram = "%s %s"%(w_i1, w_i)
            if w_i not in probs:
                probs[w_i] = 0
            if n_gram not in probs:
                probs[n_gram] = 0

            P1 = lambda_1*probs[w_i] + (1 - lambda_1)/V # probability of smoothed 1-gram
            P2 = lambda_2*probs[n_gram] + (1 - lambda_2)*P1 # probability of smoothed 2-gram
            H += -math.log(P2, 2)
            W += 1

    H = H/W

    return(H)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_file', default="output/wiki-en-train_bigram.model", help='input model file')
    parser.add_argument('-t', '--test', dest='test_file', default="../data/wiki-en-test.word", help='input test data')
    args = parser.parse_args()
    entroty = test_bigram(args.model_file, args.test_file)
    print("entropy = %f"%(entroty))