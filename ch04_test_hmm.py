# coding: utf-8
import argparse
import math
from collections import defaultdict

'''
# evaluation with this script:
$ cd ..
$ script/gradepos.pl data/wiki-en-test.pos myscript/output/wiki-en-test_result.pos

Accuracy: 90.77% (4142/4563)

Most common mistakes:
NNS --> NN  46
NN --> JJ   26
NNP --> NN  22
JJ --> DT   22
JJ --> NN   14
VBN --> NN  12
NN --> IN   12
NN --> DT   10
VBN --> JJ  10
NNP --> JJ  8

'''

_lambda = 0.95
vocab = defaultdict(lambda: 0)
possible_tags = defaultdict(lambda: 0)
emit = defaultdict(lambda: 0)
transition = defaultdict(lambda: 0)

def load_model(model_file):
    for line in open(model_file):
        T_or_E, context, word, prob = line.strip().split()
        possible_tags[context] = 1  # We use this to enumerate all tags
        if T_or_E is "T":
            transition["%s %s" % (context, word)] = float(prob)
            possible_tags[word] = 1
        else:
            emit["%s %s" % (context, word)] = float(prob)
            vocab[word] = 1


def forward_step(best_score, best_edge, words):
    for i in range(len(words)):
        for prev_tag in possible_tags.keys():
            for next_tag in possible_tags.keys():
                i_prev = '%s %s' % (i, prev_tag)
                prev_next = '%s %s' % (prev_tag, next_tag)
                next_word = '%s %s' % (next_tag, words[i])
                if i_prev in best_score and prev_next in transition:

                    # HMM transition prob
                    P_T = transition[prev_next]

                    # HMM emission prob
                    P_E = _lambda*emit[next_word]+(1-_lambda)/ len(vocab)

                    score = best_score[i_prev] + (-math.log(P_T)) + (-math.log(P_E))

                    i_next = '%s %s' % (i + 1, next_tag)
                    if i_next not in best_score or best_score[i_next] > score:
                        best_score[i_next] = score
                        best_edge[i_next] = i_prev
    return best_edge


def backward_step(best_edge, words):
    tags = []
    next_edge = best_edge['%s </s>' % (len(words))]
    while next_edge != '0 <s>':
        position, tag = next_edge.split()
        tags.append(tag)
        next_edge = best_edge[next_edge]
    tags.reverse()
    return tags


def test_hmm(model_file, test_file, result_file):
    file = open(result_file, "w")
    load_model(model_file)
    for line in open(test_file):
        words = line.strip().split()
        words.append("</s>")

        best_score = defaultdict(lambda: 0)
        best_edge = defaultdict(lambda: 0)
        best_score["0 <s>"] = 0
        best_edge["0 <s>"] = None

        best_edge = forward_step(best_score, best_edge, words)
        tags_list = backward_step(best_edge, words)
        file.write(" ".join(tags_list) + '\n')
        #print(" ".join(tags_list))


if __name__ == '__main__':

    # for test
    # test_hmm("../test/05-train-answer.txt", "../test/05-test-input.txt", "output/05-test-answer.txt")

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', dest='model', default="output/wiki-en-train_hmm.model", help='input model file')
    parser.add_argument('-t', '--test', dest='test', default="../data/wiki-en-test.norm", help='input test data')
    parser.add_argument('-r', '--result', dest='result', default="output/wiki-en-test_result.pos", help='output result file')
    args = parser.parse_args()

    test_hmm(args.model, args.test, args.result)

