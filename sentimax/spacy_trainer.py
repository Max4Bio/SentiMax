#!/usr/bin/env python
from __future__ import unicode_literals, print_function

import spacy
import os
import random

from spacy_sentiws import spaCySentiWS
from spacy import displacy
from spacy.tokens import Token
from spacy.util import minibatch, compounding
from pathlib import Path

class SpacyTrainer(object):
    """
    Helperclass to train spacy NER and dependency parser
    """

    def __init__(self, output_dir):
        pass

    def train_ner(self, train_data, model=None, new_model_name="german_modified",
                  output_dir=None, n_iter=30, labels=None, test_model=False):
        """Set up the pipeline and entity recognizer, and train the new entity."""
        # training data format:
        # TRAIN_DATA = [
        #     (
        #         "Horses are too tall and they pretend to care about your feelings",
        #         {"entities": [(0, 6, LABEL)]},
        #     ),
        #     ("Do they bite?", {"entities": []}),
        # ]
        TRAIN_DATA = train_data

        random.seed(0)
        # Add entity recognizer to model if it's not in the pipeline
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "ner" not in nlp.pipe_names:
            ner = nlp.create_pipe("ner")
            nlp.add_pipe(ner)
        # otherwise, get it, so we can add labels to it
        else:
            ner = nlp.get_pipe("ner")

        [ner.add_label(label) for label in labels]  # add new entity label to entity recognizer
        optimizer = nlp.resume_training()
        move_names = list(ner.move_names)
        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
        train_losses = []
        with nlp.disable_pipes(*other_pipes):  # only train NER
            sizes = compounding(1.0, 4.0, 1.001)
            # batch up the examples using spaCy's minibatch
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                batches = minibatch(TRAIN_DATA, size=sizes)
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
                # print("Losses", losses)
                train_losses.append(losses)

        # test the trained model
        test_text = "Do you like horses?"
        doc = nlp(test_text)
        print("Entities in '%s'" % test_text)
        displacy.render(doc, style='ent', jupyter=True)

        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.meta["name"] = new_model_name  # rename model
            nlp.to_disk(output_dir)
            print("Saved model to: ", output_dir)

            if test_model:
                # test the saved model
                print("Loading from", output_dir)
                nlp2 = spacy.load(output_dir)
                # Check the classes have loaded back consistently
                assert nlp2.get_pipe("ner").move_names == move_names
                doc2 = nlp2(test_text)
                for ent in doc2.ents:
                    print(ent.label_, ent.text)

        return train_losses

    def train_dep(self, train_data, model=None, output_dir=None, n_iter=15, test_model=False):
        """Load the model, set up the pipeline and train the parser."""
        # training data format:
        # TRAIN_DATA = [
        #     (
        #         "They trade mortgage-backed securities.",
        #         {
        #             "heads": [1, 1, 4, 4, 5, 1, 1],
        #             "deps": ["nsubj", "ROOT", "compound", "punct", "nmod", "dobj", "punct"],
        #         },
        #     ),
        # ]

        TRAIN_DATA = train_data

        # add the parser to the pipeline if it doesn't exist
        # nlp.create_pipe works for built-ins that are registered with spaCy
        if "parser" not in nlp.pipe_names:
            parser = nlp.create_pipe("parser")
            nlp.add_pipe(parser, first=True)
        # otherwise, get it, so we can add labels to it
        else:
            parser = nlp.get_pipe("parser")

        # add labels to the parser
        for _, annotations in TRAIN_DATA:
            for dep in annotations.get("deps", []):
                parser.add_label(dep)

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "parser"]
        train_losses = []
        with nlp.disable_pipes(*other_pipes):  # only train parser
            optimizer = nlp.begin_training()
            for itn in range(n_iter):
                random.shuffle(TRAIN_DATA)
                losses = {}
                # batch up the examples using spaCy's minibatch
                batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
                for batch in batches:
                    texts, annotations = zip(*batch)
                    nlp.update(texts, annotations, sgd=optimizer, losses=losses)
                # print("Losses", losses)
                train_losses.append(losses)

        # test the trained model
        test_text = "I like securities."
        doc = nlp(test_text)
        print("Dependencies", [(t.text, t.dep_, t.head.text) for t in doc])

        # save model to output directory
        if output_dir is not None:
            output_dir = Path(output_dir)
            if not output_dir.exists():
                output_dir.mkdir()
            nlp.to_disk(output_dir)
            print("Saved model to", output_dir)

            if test_model:
                # test the saved model
                print("Loading from", output_dir)
                nlp2 = spacy.load(output_dir)
                doc = nlp2(test_text)
                print("Dependencies", [(t.text, t.dep_, t.head.text) for t in doc])

        return train_losses


if __name__ == '__main__':
    pass
