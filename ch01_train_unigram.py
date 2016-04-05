# coding: utf-8
from collections import defaultdict
import math


def train_unigram(train_file):
    total_count = 0
    prob_dict = defaultdict(lambda: 0)
    word_count = defaultdict(lambda: 0)
    for line in open(train_file):
        item = line.strip().split()
        item.append("</s>")
        for word in item:
            word_count[word] += 1
            total_count += 1

    for word, count in word_count.items():
        prob_dict[word] = count/total_count


    return prob_dict


def save_file(prob_dict, model_file):
    file = open(model_file, "w")
    for word, prob in prob_dict.items():
        file.write("%s\t%f\n" % (word, prob))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train_file', default="../data/wiki-en-train.word", help='input training data')
    parser.add_argument('-m', '--model', dest='model_file', default="output/wiki-en-train_unigram.model", help='writing model file')
    args = parser.parse_args()
    prob_dict = train_unigram(args.train_file)
    save_file(prob_dict, args.model_file)

    '''
    # test
    prob_dict = train_unigram("../test/01-train-input.txt")
    save_file(prob_dict, "output/01-train-answer.txt")
    '''