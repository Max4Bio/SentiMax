#!/usr/bin/env python
import spacy

from spacy import displacy
from IPython.display import clear_output


class Entitizer(object):
    """
    Find Named-Entities in german texts
    as keyword generator for sentimax
    """

    def __init__(self, wordvecs=False):
        # loading german spacy model
        if wordvecs:
            self.nlp = spacy.load('de_core_news_md')
        else:
            self.nlp = spacy.load('de_core_news_sm')
        self.doc = None

    def find_entities(self, text, visualize=False):
        """
        Analyze text for named-entities and display it
        via IPython display in Browser

        Parameters:
        - text: some string to analyze
        - visualize: True or False; trigger for IPython display
        """
        self.doc = self.nlp(text)
        if visualize:
            displacy.render(self.doc, style='ent', page=True)
        entities = [token for token in self.doc if token.ent_type_]
        return entities

    def annotate_entities(self, texts_to_annotate):
        """
        Helper method to annotate entities like
        explosion's prodigy core

        Parameters:
        - texts_to_annotate: some strings to annotate

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

        # helper function to format true entities in spacy train format
        def format_ents(doc):
            ents = {"entities": []}
            for token in doc:
                if token.ent_type_:
                    ents["entities"].append((token.idx, token.idx + \
                    len(token), token.ent_type_))
            return (doc.text, ents)

        # annotation loop with user input
        for i in range(len(texts_to_annotate)):
            clear_output()
            self.doc = self.nlp(texts_to_annotate[i])
            displacy.render(self.doc, style='ent', page=True)
            user_input = input("Is it wrong?: ")
            if user_input == '' or user_input == None:
                true_anns.append(format_ents(self.doc))
            elif user_input.lower() == 'y':
                false_anns.append(self.doc.text)
            elif user_input.lower() == 'b':
                i -= 2
            elif user_input.lower() == 'c' or user_input.lower() == 'q':
                break

        return true_anns, false_anns


if __name__ == '__main__':
    pass
