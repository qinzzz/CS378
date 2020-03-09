# posdata.py

from typing import List
import re


class TaggedToken:
    """
    Wrapper for a token paired with a part-of-speech
    """
    def __init__(self, word: str, tag: str):
        self.word = word
        self.tag = tag

    def __repr__(self):
        return "(%s, %s)" % (self.word, self.tag)

    def __str__(self):
        return self.__repr__()


class LabeledSentence:
    """
    Thin wrapper over a sequence of Tokens representing a sentence.
    """
    def __init__(self, tagged_tokens):
        self.tagged_tokens = tagged_tokens

    def __repr__(self):
        return repr([repr(tok) for tok in self.tagged_tokens])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.tagged_tokens)

    def get_words(self):
        return [tok.word for tok in self.tagged_tokens]

    def get_tags(self):
        return [tok.tag for tok in self.tagged_tokens]


def labeled_sent_from_words_tags(words, tags):
    return LabeledSentence([TaggedToken(w, t) for (w, t) in zip(words, tags)])


def read_labeled_sents(file):
    """
    Reads a file with word<tab>tag per line, with one blank line between sentences
    :param file:
    :return:
    """
    f = open(file)
    sentences = []
    curr_tokens = []
    for line in f:
        stripped = line.strip()
        if stripped != "":
            fields = stripped.split("\t")
            if len(fields) == 2:
                curr_tokens.append(TaggedToken(fields[0], fields[1]))
        elif stripped == "" and len(curr_tokens) > 0:
            sentences.append(LabeledSentence(curr_tokens))
            curr_tokens = []
    print("Read %i sents from %s" % (len(sentences), file))
    return sentences


class Tree:
    """
    Recursive type that defines a tree structure. Wraps a label (for the current node) and a list of children which
    are also Tree objects
    """
    def __init__(self, label: str, children=[]):
        self.label = label
        self.children = children

    def __repr__(self):
        if self.is_preterminal():
            return "(%s %s)" % (self.label, self.children[0].label)
        else:
            return "(%s %s)" % (self.label, " ".join([repr(c) for c in self.children]))

    def __str__(self):
        return self.__repr__()

    def is_terminal(self):
        return len(self.children) == 0

    def is_preterminal(self):
        return len(self.children) == 1 and self.children[0].is_terminal()

    def render_pretty(self):
        return self._render_pretty_helper(0)

    def _render_pretty_helper(self, indent_level):
        if self.is_terminal():
            return (" " * indent_level) + self.label
        if self.is_preterminal():
            return (" " * indent_level) + "(" + self.label + " " + self.children[0].label + ")"
        else:
            return (indent_level * " ") + "(" + self.label + "\n" + "\n".join([c._render_pretty_helper(indent_level + 2) for c in self.children])

    def set_children(self, new_children):
        self.children = new_children

    def add_child(self, child):
        self.children.append(child)


def _read_tree(line: str) -> Tree:
    """
    :param line: a PTB-style bracketed representation of a string, like this: ( (S ... ) ) or (ROOT (S ... ) )
    :return: the Tree object
    """
    # Put in an explicit ROOT symbol for the root to make parsing easier
    raw_line = line
    if line.startswith("( "):
        line = "(ROOT " + line[1:]
    # Surround ( ) with spaces so splitting works
    line = re.sub(r"\(", " ( ", line)
    line = re.sub(r"\)", " ) ", line)
    # We may have introduced double spaces, so collapse these down
    line = re.sub(r"\s{2,}", " ", line)
    tokens = list(filter(lambda tok: len(tok) > 0, line.split(" ")))
    # Parse the bracket representation into a tree
    just_saw_open = False
    stack = []
    latest_word = ""
    for tok in tokens:
        if tok == "(":
            just_saw_open = True
        elif tok == ")":
            if latest_word != "":
                tree = stack[-1]
                tree.add_child(Tree(latest_word, []))
                latest_word = ""
                stack[-2].add_child(tree)
                stack = stack[:-1]
            else:
                if len(stack) >= 2: # only violated for the last paren
                    stack[-2].add_child(stack[-1])
                    stack = stack[:-1]
        else:
            if just_saw_open:
                tree = Tree(tok, [])
                stack.append(tree)
                just_saw_open = False
            else:
                # The only time we get a string *not* after an open paren is when it's a word and not a tag
                latest_word = tok
    if len(stack) != 1:
        print("WARNING: bad line: %s" % raw_line)
    return stack[0]


def read_parse_data(file):
    """
    :param file: a file in PTB format, one tree per line
    :return: A list of Trees
    """
    f = open(file)
    trees = []
    for line in f:
        stripped = line.strip()
        trees.append(_read_tree(stripped))
        if len(trees) % 10000 == 0:
            print("Read %i trees" % len(trees))
    return trees


if __name__=="__main__":
    trees = read_parse_data("data/alltrees_dev.mrg.oneline")
    for i in range(30, 50):
        print("==========================")
        print(trees[i].render_pretty())
