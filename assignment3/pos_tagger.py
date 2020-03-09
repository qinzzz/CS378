# pos_tagger.py

import argparse
import sys
import time
from treedata import *
from models import *

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='pos_tagger.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD or HMM)')
    parser.add_argument('--use_beam', dest='use_beam', default=False, action='store_true', help='use beam search instead of Viterbi')
    parser.add_argument('--train_path', type=str, default='data/train_sents.conll', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/dev_sents.conll', help='path to dev set (you should not need to modify)')
    args = parser.parse_args()
    return args


def print_evaluation(gold, pred):
    """
    Prints accuracy comparing gold tagged sentences and pred tagged sentences
    :param gold:
    :param pred:
    :return:
    """
    num_correct = 0
    num_total = 0
    for (gold_sent, pred_sent) in zip(gold, pred):
        for (gold_tag, pred_tag) in zip(gold_sent.get_tags(), pred_sent.get_tags()):
            num_total += 1
            if gold_tag == pred_tag:
                num_correct += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    train_sents = read_labeled_sents(args.train_path)
    dev_sents = read_labeled_sents(args.dev_path)

    # Here's a few sentences...
    print("Examples of sentences:")
    print(str(dev_sents[1]))
    print(str(dev_sents[3]))
    print(str(dev_sents[5]))
    system_to_run = args.model
    # Train our model
    if system_to_run == "BAD":
        bad_model = train_bad_tagging_model(train_sents)
        dev_decoded = [bad_model.decode(dev_ex.get_words()) for dev_ex in dev_sents]
    elif system_to_run == "HMM":
        hmm_model = train_hmm_model(train_sents)
        dev_decoded = []
        decode_start = time.time()
        for dev_ex in dev_sents:
            if args.use_beam:
                dev_decoded.append(hmm_model.beam_decode(dev_ex.get_words()))
            else:
                dev_decoded.append(hmm_model.viterbi_decode(dev_ex.get_words()))
            if len(dev_decoded) % 100 == 0:
                print("Decoded %i" % (len(dev_decoded)))
        print("Total time to tag the development set: %i seconds" % (time.time() - decode_start))
    else:
        raise Exception("Pass in either BAD, HMM, or CRF to run the appropriate system")
    # Print the evaluation statistics
    print_evaluation(dev_sents, dev_decoded)
