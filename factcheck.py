# factcheck.py
# test
import gc
import numpy as np
from collections import Counter
import spacy
from typing import List
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
import string
import torch
import re

nlp = spacy.load("en_core_web_sm")

nltk.download('punkt')
nltk.download('stopwords')
class FactExample:
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.
        # raise Exception("Don't call me, call my subclasses")

        # This is where the code should go
        predicted_label_idx = torch.argmax(logits, dim=1).item()

        if predicted_label_idx == 0:  # "entailment"
            result = "supported"
        elif predicted_label_idx == 1:  # "neutral"
            result = "neutral"
        else:  # "contradiction"
            result = "contradiction"
        
        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        return result

class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(object):
    def preprocess_text(self, text: str):
        # Tokenize the text and remove punctuation
        doc = nlp(text)
        tokens = word_tokenize(text.replace("<s>", "").replace("</s>", ""))
        # tokens = [word for word in tokens if word not in punctuation_to_remove]
        tokens = [word if not re.match(r'^\d+$', word) else 'NUM' for word in tokens]
        tokens = [word.lower() for word in tokens if word != 'NUM' and word.isalpha() and word not in string.punctuation]

        # Remove stop words
        # stop_words = set(stopwords.words('english'))
        stop_words = ["the", "and", "is", "to", "a", "of", "in", "it", "I", "that", "you", "he", "she", "we", "they"]
        tokens = [word for word in tokens if word not in stop_words]

        # Stem the remaining words
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)
    
    def predict(self, fact: str, passages: List[dict]) -> str:
        fact = self.preprocess_text(fact)
        fact_word_set = set(fact.split())
        
        supported = False
        for passage in passages:
            passage_text = passage['text']
            passage_text = self.preprocess_text(passage_text)
            passage_word_set = set(passage_text.split())
            
            intersection = len(fact_word_set.intersection(passage_word_set))
            jaccard_sim = intersection / len(fact_word_set)

            if jaccard_sim > 0.7:
                supported = True

        if supported:
            return "S"
        else:
            return "NS"


class EntailmentFactChecker(object):
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))

    def preprocess_text(self, text: str) -> List[str]:
        # Tokenize
        tokens = word_tokenize(text.replace("<s>", "").replace("</s>", ""))
        tokens = [word if not re.match(r'^\d+$', word) else 'NUM' for word in tokens]
        tokens = [word.lower() for word in tokens if word !='NUM' and word.isalpha() and word not in string.punctuation]
        return [self.stemmer.stem(token) for token in tokens if token.lower() not in self.stopwords and token.isalpha()]
    
    # same as part 1, but threshold needed to be tweaked
    def word_overlap(self, fact: str, passages: List[dict]) -> str:
        fact_tokens = set(self.preprocess_text(fact))
        for passage in passages:
            passage_tokens = set(self.preprocess_text(passage['text']))
            intersection = fact_tokens.intersection(passage_tokens)
            jaccard = len(intersection) / len(fact_tokens)
            if jaccard > 0.45:
                return "S"
        return "NS"

    def predict(self, fact: str, passages: List[dict]) -> str:
        neutral_count = 0
        if self.word_overlap(fact, passages) == 'S':
            for passage in passages:
                sentences = sent_tokenize(passage['text'])
                for sentence in sentences:
                    total_sentences = len(sentences) * len(passages)
                    # get entailment result
                    result = self.ent_model.check_entailment(
                        sentence, fact)
                    # return early is supported
                    # once one sentence is supported, the fact is supported
                    if result == 'supported':
                        return "S"
                    # we dont know yet
                    if result == 'neutral':
                        neutral_count += 1
                        # if all neutral, then we are saying its supported
                        if neutral_count == total_sentences:
                            return "S"
        # if it hasnt returned supported yet, then it is not supported
        return 'NS'


# OPTIONAL
class DependencyRecallThresholdFactChecker(object):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations

