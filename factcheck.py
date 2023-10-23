# factcheck.py
import numpy as np
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string


nlp = spacy.load("en_core_web_sm")

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

    def check_entailment(self, premise: str, hypothesis:str ):
        # Tokenize the premise and hypothesis
        inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
        # Get the model's prediction
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Note that the labels are ["contradiction", "neutral", "entailment"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.
        raise Exception("Not implemented")


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
        tokens = word_tokenize(text)
        tokens = [word.lower() for word in tokens if word.isalpha() and word not in string.punctuation and word != '<s>']

        # Remove stop words
        stop_words = set(stopwords.words('english'))
        # stop_words = ["the", "and", "is", "to", "a", "of"]
        tokens = [word for word in tokens if word not in stop_words]

        # Stem the remaining words
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)
    
    def jaccard_similarity(self, set1, set2):
        intersection = len(set1.intersection(set2))
        union = len(set1) + len(set2) - intersection
        return intersection / union if union != 0 else 0  # To avoid division by zero

    def predict(self, fact: str, passages: List[dict]) -> str:
        print("Fact: ", fact)

        # Preprocess fact
        fact = self.preprocess_text(fact)
        fact_word_set = set(fact.split())

        # Initialize lists to store Jaccard similarity scores
        similarity_scores = []

        for passage in passages:
            passage_text = passage['text']

            # Preprocess passage text
            passage_text = self.preprocess_text(passage_text)
            passage_word_set = set(passage_text.split())

            # Calculate Jaccard similarity between fact and passage
            jaccard_sim = self.jaccard_similarity(fact_word_set, passage_word_set)
            similarity_scores.append((passage['title'], jaccard_sim))

        supported = False

        # Print the similarity scores for each passage
        for title, score in similarity_scores:
            if score > 0.02:  # You can adjust the threshold here
                supported = True
            print(f"Jaccard Similarity between Fact and '{title}': {score}")

        if supported:
            return "S"
        else:
            return "NS"

    # cosine similarity
    # def predict(self, fact: str, passages: List[dict]) -> str:
    #     print("Fact: ", fact)

    #     # Preprocess fact
    #     fact = self.preprocess_text(fact)

    #     # Initialize lists to store cosine similarity scores
    #     similarity_scores = []

    #     # Initialize TF-IDF vectorizer
    #     tfidf_vectorizer = TfidfVectorizer()

    #     # Tokenize and create TF-IDF vectors for each passage, then calculate cosine similarity
    #     for passage in passages:
    #         passage_text = passage['text']

    #         # Preprocess passage text
    #         passage_text = self.preprocess_text(passage_text)

    #         # Create TF-IDF vectors for fact and passage text
    #         tfidf_matrix = tfidf_vectorizer.fit_transform([fact, passage_text])
    #         cosine_sim = cosine_similarity(tfidf_matrix)

    #         # Cosine similarity between fact and passage
    #         similarity_score = cosine_sim[0, 1]
    #         similarity_scores.append((passage['title'], similarity_score))

    #     supported = False
    #     # Print the similarity scores for each passage
    #     for title, score in similarity_scores:
    #         if score > 0.03:  # You can adjust the threshold here
    #             supported = True
    #         print(f"Cosine Similarity between Fact and '{title}': {score}")

    #     if supported:
    #         return "S"
    #     else:
    #         return "NS"


        


class EntailmentFactChecker(object):
    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


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