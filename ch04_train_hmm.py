# coding: utf-8
import argparse
from collections import defaultdict

context = defaultdict(lambda: 0)
emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)

def train_hmm(train_file):
    for line in open(train_file):
        words = line.strip().split()
        previous_tag = "<s>"    # Make the sentence start
        context[previous_tag] += 1
        for word_tag in words:
            word, tag = word_tag.split("_")
            context[tag] += 1  # Count the context
            emit["%s %s" % (tag, word)] += 1   # Count the emission
            if previous_tag is not "":
                transition["%s %s" % (previous_tag, tag)] += 1  # Count the transition with bigram data
            previous_tag = tag
        transition[previous_tag+" </s>"] += 1

def save_file(model_file):
    file = open(model_file, "w")
    for key, value in sorted(transition.items()):
        previous, word = key.split()
        file.write('T %s %f\n' % (key, value / context[previous]))
    for key, value in sorted(emit.items()):
        previous, word = key.split()
        file.write('E %s %f\n' % (key, value / context[previous]))


if __name__ == '__main__':

    # test
    #train_hmm("../test/05-train-input.txt")
    #save_file("output/05-train-answer.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', dest='train', default="../data/wiki-en-train.norm_pos", help='input trainging data')
    parser.add_argument('-m', '--model', dest='model', default="output/wiki-en-train_hmm.model", help='output model file')
    args = parser.parse_args()

    train_hmm(args.train)
    save_file(args.model)



