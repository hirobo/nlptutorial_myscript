# coding: utf-8
from collections import defaultdict
import math

def load_model(model_file):
    probs = defaultdict(lambda: 0)
    for line in open(model_file):
        #print(line)
        word, probability = line.strip().split("\t")
        probs[word] = float(probability)
    return probs

def calc_probabilitiy(word, probs):
    # calc unigram probability
    N = 10**6
    lambda_1 = 0.95
    probability = (1 - lambda_1)/N
    if word in probs:
        probability += lambda_1*probs[word]
    return probability

def test_unigram(model_file, test_file):
    H = 0 # entropy
    W = 0 # word count
    unk = 0  # num of unknown words

    probs = load_model(model_file)
    for line in open(test_file, "r"):
        words = line.strip().split()
        words.append("</s>")
        for word in words:
            W += 1
            P = calc_probabilitiy(word, probs)
            H += -math.log(P, 2)
            if word not in probs:
                unk += 1
    H = H/W

    return(H, (W-unk)/W)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model_file', default="output/wiki-en-train_unigram.model", help='input model file')
    parser.add_argument('-t', '--test', dest='test_file', default="../data/wiki-en-test.word", help='input test data')
    args = parser.parse_args()
    entroty, coverage = test_unigram(args.model_file, args.test_file)
    print("entropy = %f"%(entroty))
    print("coverage = %f"%(coverage))

    # this should be same as ../test/01-train-answer.txt
    # entroty, coverage = test_unigram("output/01-train-answer.txt", "../test/01-test-input.txt")
    # print("entropy = %f"%(entroty))
    # print("coverage = %f"%(coverage))
