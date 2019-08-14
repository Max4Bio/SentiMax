#!/usr/bin/env python
from __future__ import unicode_literals, print_function

import spacy
import os
import pickle
import pandas as pd
import random

from spacy_sentiws import spaCySentiWS
from spacy import displacy
from spacy.tokens import Token
from spacy.util import minibatch, compounding
from pathlib import Path
from IPython.display import clear_output


class SentiMax(object):
    """
    Analyze german text for determining polarity values of
    several sentiment associated tokens. The polarity values are calculated
    with the german SentiWS-Corpus, enhancing & silencing tokens (* 1.5 / * 0.5)
    and negations (* -1.0)
    """

    def __init__(self, wordvecs=False, sentiws_path='data/sentiws/',
                 polarity_modifiers_path='data/polarity_modifiers.pickle'):
        """
        Parameters:
        - wordvecs: True or False; use de_core_news_sm
                    or de_core_news_md german spacy model
        - sentiws_path: path of your sentiws data
        - polarity_modifiers_path: path of your polarity
          modifier dict as pickle
        """
        # loading german spacy model
        if wordvecs:
            self.nlp = spacy.load('de_core_news_md')
        else:
            self.nlp = spacy.load('de_core_news_sm')
        # integrating SentiWS-Corpus as token attribute
        sentiws = spaCySentiWS(sentiws_path=sentiws_path)
        self.nlp.add_pipe(sentiws)
        self.doc = None
        self.modifiers = pickle.load(open(polarity_modifiers_path, 'rb'))
        if not Token.has_extension("modified"):
            Token.set_extension("modified", getter=self.modify_polarity)
        if not Token.has_extension("negated"):
            Token.set_extension("negated", getter=self.negate)

    def modify_polarity(self, token):
        """
        Modifies token polarity values by polarity modifiers.

        Parameters:
        - token: spacy token

        return: float
        - polarity value of token modified by
          polarity enhancer or polarity reducer
        """
        children = token.children
        polarity = token._.sentiws
        if not polarity:
            polarity = 0.0
        for child in children:
            if child.lower_ in self.modifiers['polarity_enhancer']:
                return polarity * 1.5
            elif child.lower_ in self.modifiers['polarity_reducer']:
                return polarity * 0.5
        return polarity

    def negate(self, token):
        """
        Negate the polarity of a token if there is a negation
        under its children.

        Parameters:
        - token: spacy token

        return: float
        - negated polarity value
        """
        children = token.children
        polarity = token._.modified
        for child in children:
            if child.dep_ == 'ng':
                return -1.0 * polarity
        return polarity

    def polarize(self, text):
        """
        Just creating a spacy doc from the input string
        """
        self.doc = self.nlp(text)

    def sentimax(self):
        """
        Converts dependecies and polarity values of tokens
        in a doc to as pandas DataFrame

        return: pd.DataFrame
        """
        polarity_dict = {"token": [], "dep": [], "sentiws": [], "modified": [], "negated": []}
        if self.doc:
            for token in self.doc:
                polarity_dict["token"].append(token.text)
                polarity_dict["dep"].append(token.dep_)
                polarity_dict["sentiws"].append(token._.sentiws)
                polarity_dict["modified"].append(token._.modified)
                polarity_dict["negated"].append(token._.negated)

        return pd.DataFrame(data=polarity_dict)

    def annotate_dependencies(self, texts_to_annotate, display=False):
        """
        Helper method to annotate dependencies like
        explosion's prodigy core

        Parameters:
        - texts_to_annotate: some strings to annotate
        - display: True or False

        control:
        - No input, just hit enter: Append the right
          annotated string to list of right annotated strings.
        - Y, y: Append false annotated string to list of false
          annotated strings.
        - B, b (backwards): go to last string.
        - C, c, Q, q (cancel/quit): break loop to quit and return
          true and ´false annotated strings.

        return: tuple(true_annotated, false_annotated)
                - true annotated strings are formatted to spacy train format
                - false annotated string are returned
                  as same as inputted strings
        """
        true_anns = []
        false_anns = []
        # helper function to format true dependecies in spacy train format
        def format_deps(doc):
            deps = {"heads": [], "deps": []}
            for token in doc:
                deps["heads"].append(token.head)
                deps["deps"].append(token.dep_)
            return (doc.text, deps)

        # annotation loop with user input
        for i in range(len(texts_to_annotate)):
            self.doc = self.nlp(texts_to_annotate[i])
            if display:
                clear_output()
                displacy.render(self.doc, style='dep', page=True)
            else:
                deps = []
                for token in self.doc:
                    deps.append("({}, {}, {})".format(token.text, token.dep_, token.head))
                print(" ".join(deps))
            user_input = input("Is it wrong?: ")
            if user_input == '' or user_input == None:
                true_anns.append(format_deps(self.doc))
            elif user_input.lower() == 'y':
                false_anns.append(self.doc.text)
            elif user_input.lower() == 'b':
                i -= 2
            elif user_input.lower() == 'c' or user_input.lower() == 'q':
                break

        return true_anns, false_anns


if __name__ == '__main__':
    pass
