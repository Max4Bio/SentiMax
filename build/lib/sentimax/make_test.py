#!/usr/bin/env python
import spacy
import os
import pandas as pd
import pickle

from entitizer import Entitizer
from sentimax import SentiMax
from spacy_trainer import SpacyTrainer
from IPython.display import display, HTML


if __name__ == '__main__':
	clinic_reviews = pickle.load(open("Klinikbewertungen", 'rb'))
	# test Entitizer
	entitizer = Entitizer()
	entitizer.find_entities(clinic_reviews[0], visualize=True)

	# test SentiMax
	sentimax = SentiMax()
	sentimax.polarize(clinic_reviews[0])
	df = sentimax.sentimax()
	display(HTML(df.to_html()))
