# lm.py

import argparse
import json
import time
from models import *
from utils import *

####################################################
# DO NOT MODIFY THIS FILE IN YOUR FINAL SUBMISSION #
####################################################


def _parse_args():
    """
    Command-line arguments to the system. --model switches between the main modes you'll need to use. The other arguments
    are provided for convenience.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='lm.py')
    parser.add_argument('--model', type=str, default='UNIFORM', help='model to run (UNIFORM or RNN)')
    parser.add_argument('--train_path', type=str, default='data/text8-100k.txt', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/text8-dev.txt', help='path to dev set (you should not need to modify)')
    parser.add_argument('--output_bundle_path', type=str, default='output.json', help='path to write the results json to (you should not need to modify)')
    args = parser.parse_args()
    return args


def read_text(file):
    """
    :param file:
    :return: The text in the given file as a single string
    """
    all_text = ""
    for line in open(file):
        all_text += line.strip()
    print("%i chars read in" % len(all_text))
    return all_text


def run_sanity_check(lm):
    """
    Runs a sanity check that the language model returns valid probabilities for a few contexts. Checks that your model
    can take sequences of different lengths and contexts of different lengths without crashing.
    :param lm: the trained LM
    :return: True if the output is sane, false otherwise
    """
    contexts = [" ", " a person ", " some kind of person "]
    next_seqs = ["s", "sits", "sits there"]
    sane = True
    for context in contexts:
        for next_seq in next_seqs:
            log_prob = lm.get_log_prob_sequence(next_seq, context)
            if log_prob > 0.0:
                sane = False
                print("ERROR: sanity checks failed, LM probability %f is invalid" % (log_prob))
    return sane


def print_evaluation(text, lm, output_bundle_path):
    """
    Runs both the sanity check and also runs the language model on the given text and prints three metrics: log
    probability of the text under this model (treating the text as one log sequence), average log probability (the
    previous value divided by sequence length), and perplexity (averaged "branching favor" of the model)
    :param text: the text to evaluate
    :param lm: model to evaluate
    :param output_bundle_path: the path to print the output bundle to, in addition to printing it
    """
    sane = run_sanity_check(lm)
    log_prob = lm.get_log_prob_sequence(text, " ")
    avg_log_prob = log_prob/len(text)
    perplexity = np.exp(-log_prob / len(text))
    data = {'sane': sane, 'log_prob': log_prob, 'avg_log_prob': avg_log_prob, 'perplexity': perplexity}
    print("=====Results=====")
    print(json.dumps(data, indent=2))
    with open(output_bundle_path, 'w') as outfile:
        json.dump(data, outfile)


if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)

    train_text = read_text(args.train_path)
    dev_text = read_text(args.dev_path)

    # Vocabs is lowercase letters a to z and space
    vocab = [chr(ord('a') + i) for i in range(0, 26)] + [' ']
    vocab_index = Indexer()
    for char in vocab:
        vocab_index.add_and_get_index(char)
    print(repr(vocab_index))

    print("First 100 characters of train:")
    print(train_text[0:100])
    # Train our model
    if args.model == "RNN":
        model = train_lm(args, train_text, dev_text, vocab_index)
    elif args.model == "UNIFORM":
        model = UniformLanguageModel(len(vocab))
    else:
        raise Exception("Pass in either UNIFORM or LSTM to run the appropriate system")

    print_evaluation(dev_text, model, args.output_bundle_path)
