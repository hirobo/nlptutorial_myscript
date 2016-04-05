# coding: utf-8
import math
from collections import defaultdict
import ch01_test_unigram

'''
# [Step1] create model
$ python ch01_train_unigram.py -t ../data/wiki-ja-train.word -m output/wiki-ja-train_unigram.model

# [Step2] devide words with this script
$ python ch03_word_devide_viterbi.py

# [Step3] evaluate with this script
$ cd ..
$ script/gradews.pl data/wiki-ja-test.word myscript/output/ch03_wiki-ja-test_result.word

# Result
Sent Accuracy: 23.81% (20/84)
Word Prec: 68.93% (1861/2700)
Word Rec: 80.77% (1861/2304)
F-meas: 74.38%
Bound Accuracy: 83.25% (2683/3223)
'''
def word_devide(model_file, test_file, result_file=None):
    # load model
    unigram_dict = ch01_test_unigram.load_model(model_file)

    result = []
    for line in open(test_file, 'r'):
        line = line.strip()
        # forward viterbi
        best_edge, best_score = forward_step(line, unigram_dict)
        # backward viterbi
        words = backward_step(line, best_edge, best_score)
        result.append(" ".join(words))

    # return result if result_file is not assigned
    if result_file is None:
        return result

    else:
        file = open(result_file, "w")
        for item in result:
            file.write("%s\n"%item)

        print("devided words saved on: %s"%result_file)

def forward_step(line, unigram_dict):
    # forward viterbi
    best_edge = defaultdict(lambda: 0)
    best_score = defaultdict(lambda: 0)

    best_edge[0] = None
    best_score[0] = 0

    for word_end in range(1, len(line)+1):
        best_score[word_end] = float("inf")
        best_edge[word_end] = None
        for word_begin in range(0, word_end):
            word = line[word_begin:word_end]

            if(word in unigram_dict or len(word) == 1):
                #prob = unigram_dict[word]
                prob = ch01_test_unigram.calc_probabilitiy(word, unigram_dict)
                my_score = best_score[word_begin] - math.log(prob)
                if(my_score < best_score[word_end]):
                    best_score[word_end] = my_score
                    best_edge[word_end] = (word_begin, word_end)

    return (best_edge, best_score)

def backward_step(line, best_edge, best_score):
    words = []
    next_edge = best_edge[len(best_edge) - 1]
    while next_edge != None:
        word = line[next_edge[0]:next_edge[1]]
        words.append(word)
        next_edge = best_edge[next_edge[0]]

    words = words[::-1] # reverse

    return words

if __name__ == "__main__":
    import argparse

    # print("[Test 1]-----------------") #this output should be same as "../test/04-answer.txt"
    # res = word_devide("../test/04-model.txt", '../test/04-input.txt', None)

    # # output result
    # for item in res:
    #     print(item)

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', dest='model', default="output/wiki-ja-train_unigram.model", help='input model data')
    parser.add_argument('-t', '--test', dest='test', default="../data/wiki-ja-test.txt", help='input test data')
    parser.add_argument('-r', '--result', dest='result', default="output/ch03_wiki-ja-test_result.word", help='output result file')
    args = parser.parse_args()
    word_devide(args.model, args.test, args.result)