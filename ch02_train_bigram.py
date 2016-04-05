# coding: utf-8
from collections import defaultdict

def train_bigram(train_file):
    prob_dict = {}

    # load training data from file
    counts = defaultdict(lambda: 0)
    context_counts = defaultdict(lambda: 0)

    # train
    for line in open(train_file, "r"):
        words = line.strip().split()
        words.insert(0, '<s>')
        words.append("</s>")
        for i in range(1, len(words)): # beggins after <s>
            n_gram = "%s %s"%(words[i-1], words[i])
            # add for 2-gram
            counts[n_gram] += 1
            context_counts[words[i-1]] += 1
            # add for 1-gram
            counts[words[i]] += 1
            context_counts[""] += 1

    for n_gram, value in counts.items():
        l = n_gram.split(" ")[:-1] #remove the last element. [w_{i-1}, w_{i}] => [w_{i-1}]
        context = "".join(l)
        prob_dict[n_gram] = value/context_counts[context]

    return prob_dict


def save_file(prob_dict, model_file):
    file = open(model_file, "w")
    for word, prob in prob_dict.items():
        file.write("%s\t%f\n" % (word, prob))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train_file', default="../data/wiki-en-train.word", help='input training data')
    parser.add_argument('-m', '--model', dest='model_file', default="output/wiki-en-train_bigram.model", help='writing model file')
    args = parser.parse_args()
    prob_dict = train_bigram(args.train_file)
    save_file(prob_dict, args.model_file)

    #'''
    # test
    prob_dict = train_bigram("../test/02-train-input.txt")
    save_file(prob_dict, "output/02-train-answer.txt")
    #'''