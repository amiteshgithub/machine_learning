"""COVID-19 Information chat bot"""

import os
import re
import time
import sys
import inspect
import random

import contractions
import nltk
import spacy
import spacy_universal_sentence_encoder
import ssl
import swifter
import pandas as pd

from autocorrect import Speller
from bs4 import BeautifulSoup
from colorama import init
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer

from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tag.util import untag
from pyfiglet import figlet_format
from spacy import displacy
from termcolor import cprint

# Had to add below try statement (Otherwise getting error while trying to download from nltk)
try:

    _create_unverified_https_context = ssl._create_unverified_context

except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
spell = Speller()
token = ToktokTokenizer()
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
charac = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~0123456789'
stop_words = set(stopwords.words("english"))
adjective_tag_list = set(['JJ','JJR', 'JJS', 'RBR', 'RBS']) # List of Adjective's tag from nltk package

########################################################################################
# For sentence conversions
CONTRACTIONS_DICT = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

########################################################################################
# For logging
import logging

# create logger
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
log.addHandler(ch)
log.propagate = False
########################################################################################

# Python Question and Answers csvs
QUESTIONS_DATA = ('%s' % (os.path.join(os.path.dirname(__file__), './Questions.csv')))
ANSWERS_DATA = ('%s' % (os.path.join(os.path.dirname(__file__), './Answers.csv')))
TAGS_DATA = ('%s' % (os.path.join(os.path.dirname(__file__), './Tags.csv')))
CLEAN_QUESTIONS_CSV_PATH = (
                '%s' % (os.path.join(os.path.dirname(__file__), './df_questions_fullclean.csv'))
        )
########################################################################################


class PythonFAQChatbot(object):
    """Python FAQ chat bot """
    dtypes_questions = {'Id': 'int32', 'Score': 'int16', 'Title': 'str', 'Body': 'str'}

    def __init__(self):
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        # set df_tags
        self.df_tags = pd.read_csv(TAGS_DATA)

        # Sets dafa frame for Python Questions
        if (os.path.exists(CLEAN_QUESTIONS_CSV_PATH)):
            self.df = pd.read_csv(CLEAN_QUESTIONS_CSV_PATH)
        else:
            self.set_dafa_frame_from_csv(QUESTIONS_DATA)
            # setup df questions
            self.setup_df()

        # Set df for answers csv
        self.df_answers = pd.read_csv(
            ANSWERS_DATA,
            encoding="ISO-8859-1",
            dtype=self.dtypes_questions
        )
        self.df_answers = self.df_answers[
            self.df_answers['ParentId'].isin(
                list(self.df_tags[self.df_tags['Tag'].isin(['Python', 'python'])]['Id'])
            )
        ]

        # Load model
        self.load_model()

        # Create clean questions column
        if (not os.path.exists(CLEAN_QUESTIONS_CSV_PATH)):
            self.create_clean_text_column()

        log.debug(f"{self.df[['Title', 'Body', 'clean_text']]}")

        self.nlp_for_user_question = spacy.load('en_core_web_lg')

        # Sample question (to create similarity column) - > next time onwards it gets fast
        self.get_answer_for_most_similar_title_plus_body('How to add two lists')
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def set_dafa_frame_from_csv(self, csv):
        """Sets data frame based on the csvs passed

        :param string csv: path of csv file

        :returns pd.dataFrame: sets data frame based on the csv passed

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')

        # These are the relevant columns needed for chat bot (Rest I will remove)
        # And these are the common columns available in all csvs
        columns_to_keep = ['Id', 'Score', 'Title', 'Body']

        self.df = pd.read_csv(
            csv,
            usecols=columns_to_keep,
            encoding="ISO-8859-1",
            dtype=self.dtypes_questions
        )
        self.df = self.df[
            self.df['Id'].isin(
                list(self.df_tags[self.df_tags['Tag'].isin(['Python', 'python'])]['Id'])
            )
        ]
        self.df = self.df.sort_values(by=['Id'])
        self.df.dropna(inplace=True)
        self.df = self.df.reset_index(drop=True)

        log.debug(f'shape:{self.df.shape}')
        log.debug(f'df columns:{self.df.columns.to_list()}')
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def load_model(self):
        """Loads the model"""
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        # This library lets you use Universal Sentence Encoder embeddings of Docs, Spans and Tokens
        # directly from TensorFlow Hub
        self.nlp_for_sent_similarity = spacy_universal_sentence_encoder.load_model('en_use_lg')
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')


    def _encode_decode_df_columns(self, columns, encoding='utf-8', decoding='ISO-8859-1'):
        self.df[columns] = self.df[columns].applymap(
            lambda x: str(x).encode(encoding, errors='surrogatepass').decode(
                decoding,
                errors='surrogatepass'
            )
        )

    def clean_sentence(self, text, lemmatize=True):
        """Clean the sentence

        :param string text: text to be cleaned
        :param boolean lemmatize: lemmatize sentence if True else Don't

        :returns string: clean sentence

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')

        # Cleaning sentence
        text = str(text).lower()
        text = re.sub(r'\([^)]*\)', '', text)
        text = re.sub('"', '', text)
        text = ' '.join([CONTRACTIONS_DICT[t] if t in CONTRACTIONS_DICT else t for t in
                              text.split(" ")])

        # Removing all non-alphabetical character
        text = re.sub(r"'s\b", "", text)
        text = re.sub("[^a-zA-Z]", " ", text)

        # # match all literal apostrophe pattern then replace them by a single whitespace
        # text = re.sub(r"\'", "'", text)
        # # match all literal Line Feed (New line) pattern then replace them by a single whitespace
        # text = re.sub(r"\n", " ", text)
        # # match all literal non-breakable space pattern then replace them by a single whitespace
        # text = re.sub(r"\xa0", " ", text)
        # # match all one or more whitespace then replace them by a single whitespace
        # text = re.sub('\s+', ' ', text)
        # text = text.strip(' ')
        #
        # # Remove contractions
        # text = contractions.fix(text)
        #
        # words = token.tokenize(text)
        # words_correct = [spell(w) for w in words]
        # text = ' '.join(map(str, words_correct))
        #
        # # Lower the text again
        # text = text.lower()
        #
        # # Removing all non-alphabetical character
        # text = re.sub(r"\b\w{1}\b", "", text)  # remove all single letter
        # text = re.sub("\s+", " ", text)  # remove whitespaces left after the last operation
        # text = text.strip(" ")
        #
        # # remove single alphabetical character
        # text = re.sub(r"\b\w{1}\b", "", text)  # remove all single letter
        # text = re.sub("\s+", " ", text)  # remove whitespaces left after the last operation
        # text = text.strip(" ")
        #
        # # remove common words in english by using nltk.corpus's list
        # words = token.tokenize(text)
        # filtered = [w for w in words if not w in stop_words]
        # text = ' '.join(map(str, filtered))
        #
        # # remove all words by using ntk tag (adjectives, verbs, etc.)
        # words = token.tokenize(text)  # Tokenize each words
        # words_tagged = nltk.pos_tag(tokens=words, tagset=None, lang='eng')  # Tag each words and return a list of tuples (e.g. ("have", "VB"))
        # filtered = [w[0] for w in words_tagged if w[1] not in adjective_tag_list]  # Select all words that don't have the undesired tags
        # text = ' '.join(map(str, filtered))  # text untokenize
        #
        # # Stem the text
        # words = nltk.word_tokenize(text)  # tokenize the text then return a list of tuple (token, nltk_tag)
        # stem_text = []
        # for word in words:
        #     stem_text.append(stemmer.stem(word))  # Stem each words
        # text = " ".join(stem_text)  # Return the text untokenize
        #
        # # if lemmatize True then lemmatize sentence
        # if (lemmatize):
        #     # Lemmatize the text by using tag
        #     tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))  # tokenize the text then return a list of tuple (token, nltk_tag)
        #     lemmatized_text = []
        #     for word, tag in tokens_tagged:
        #         if tag.startswith('J'):
        #             lemmatized_text.append(lemmatizer.lemmatize(word, 'a'))  # Lemmatisze adjectives. Not doing anything since we remove all adjective
        #         elif tag.startswith('V'):
        #             lemmatized_text.append(lemmatizer.lemmatize(word, 'v'))  # Lemmatisze verbs
        #         elif tag.startswith('N'):
        #             lemmatized_text.append(lemmatizer.lemmatize(word, 'n'))  # Lemmatisze nouns
        #         elif tag.startswith('R'):
        #             lemmatized_text.append(lemmatizer.lemmatize(word, 'r'))  # Lemmatisze adverbs
        #         else:
        #             lemmatized_text.append(lemmatizer.lemmatize(word))  # If no tags has been found, perform a non specific lemmatization
        #     text = " ".join(lemmatized_text)  # Return the text untokenize

        words = word_tokenize(text)

        # For reducing words to their root form
        lemma = WordNetLemmatizer()

        # if lemmatize True then lemmatize sentence and remove stopwords
        if (lemmatize):
            stop_words = set(stopwords.words('english'))
            words = [lemma.lemmatize(word, 'v') for word in words if (word) not in stop_words]

        return  (" ".join(words)).strip()

    def sentence_similarity(self, sentence_1, sentence_2):
        """Get similarity between two sentences

        :param string sentence_1: sentence 1
        :param string sentence_1: sentence 2

        :returns float: similarity index

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        # USed to get similarity between user question and covid-19 database questions
        sentence_1 = self.nlp_for_sent_similarity(sentence_1)
        sentence_2 = self.nlp_for_sent_similarity(sentence_2)
        return sentence_1.similarity(sentence_2)

    def setup_df(self):
        # Encode decode df columns 'Title' and 'Body'
        self._encode_decode_df_columns(['Title', 'Body'], encoding='utf-8', decoding='ISO-8859-1')

        # Remove all questions that have a negative score
        self.df = self.df[self.df['Score'] >= 0]

    def create_clean_text_column(self):
        """Create a new column in dataframe 'clean_text' with concatenated cleaned text from columns
         'Title' and 'Body'

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')

        # Parse body and title then covert to text using BeautifulSoup
        self.df['Body'] = self.df['Body'].apply(
            lambda x: BeautifulSoup(x, 'html.parser').get_text()
        )
        self.df['Title'] = self.df['Title'].apply(
            lambda x: BeautifulSoup(x, 'html.parser').get_text()
        )

        self.df['clean_title'] = self.df['Title'].swifter.apply(self.clean_sentence)
        self.df['clean_body'] = self.df['Body'].swifter.apply(self.clean_sentence)
        self.df['clean_text'] = self.df['clean_title'] + ' ' + self.df['clean_body']
        # Save clean_questions csv
        self.df.to_csv(CLEAN_QUESTIONS_CSV_PATH, encoding='utf-8', errors='surrogatepass')

        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def create_sentence_similarity_column(self, user_question):
        """Create a new column in dataframe 'sim' that will correspond to similarity between user
        question and dataframe questions (using column 'clean_text') for comparison. This similarity
        will be used to later help in algorithm to select answer best matched for user question

        :param string user_question: user question

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        self.df['sim'] = ''
        user_question = self.clean_sentence(user_question)
        self.df['sim'] = self.df['clean_text'].swifter.apply(
            self.sentence_similarity,
            args=(user_question,)
        )
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def get_nouns_verbs_adj_from_user_question(self, user_question):
        """Returns all nouns, verbs and adjectives in user question

        :param string user_question: user question

        :returns list: nouns, verbs and adjectives in user question

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        user_question = self.clean_sentence(user_question)
        lemma = WordNetLemmatizer()
        noun_verb_list = [
            ent.text for ent in self.nlp_for_user_question(user_question) if (ent.pos_ in ['NOUN', 'VERB', 'ADJ'])
        ]
        return [lemma.lemmatize(word, 'v') for word in noun_verb_list]

    def check_user_question_nouns_in_df_answer_and_question(self, user_question_nouns, Id):
        """Checks if user question's nouns, verbs and adjective in 'clean_answer' and
        'clean_question' column at index 'index' in df

        :param list user_question_nouns: nouns, verbs and adjectives in user question
        :param int index: Id for given question df

        :returns boolean: True if any of the nouns, verbs and adjectives found in clean_answer' or
        'clean_question' column at index 'index' in df else False

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        log.debug(f"clean_answer: {self.df._get_value(int(index), 'clean_answer')}, index: {index}")
        log.debug(
            f"clean_question: {self.df._get_value(int(index), 'clean_question')}, index: {index}"
        )
        if (any([noun in self.df._get_value(int(index) ,'clean_answer') for noun in user_question_nouns])
                or any([noun in self.df._get_value(int(index), 'clean_question') for noun in user_question_nouns])):
            return True

        return False

    def get_answer_for_most_similar_title_plus_body(self, user_question):
        """Checks the dataframe and looks for most similar question in df to user question

        :param string user_question: user question

        :returns string: best suited answer for user's question

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        log.debug('Creating question similarity column')
        self.create_sentence_similarity_column(user_question)

        df_copied = self.df.copy()
        # sort index values based on the decreasing order of similarity (using 'sim' column we
        # created by checking similarity against user question)
        df_copied = df_copied.sort_values(by='sim', ascending=False)

        # Get sorted index values of df in a list
        sorted_index_values_based_on_sim = df_copied.index.values.tolist()

        # For debugging print below info
        log.debug(f"{df_copied[['Title', 'clean_text', 'answer', 'sim']]}")
        log.debug(f'sorted_index_values_based_on_sim: {sorted_index_values_based_on_sim}')

        # Get all nouns, verbs and adjectives in user question as a list
        user_question_nouns_verbs_adj = self.get_nouns_verbs_adj_from_user_question(user_question)
        log.debug(
            f'user_question_nouns: {self.get_nouns_verbs_adj_from_user_question(user_question)}'
        )

        # If no nouns, verbs and adjectives in user question, return Sorry...Invalid question!
        if (not user_question_nouns_verbs_adj):
            return 'Sorry...Invalid question!'


        # Now iterate df based on sorted index values and check for which index we find user
        # question's nouns, verbs and adjective in 'clean_answer' or 'clean_question' column
        try:

            df_answers = self.df_answers[
                self.df_answers['ParentId'] == self.df_questions._get_value(int(0), 'Id')
            ].sort_values(by='Score', ascending=False)
            top_anwer = df_answers._get_value(int(0) ,'Body')
            return top_anwer

        except Exception:
            # Return 'Sorry...No suitable answer available in database!' if no suitable answer found
            return 'Sorry...No suitable answer available in database!'

    # Todo: If time permits
    def print_article_summary_with_entities(self):
        pass

    def print_chatbot_ready_text(self, text):
        """Prints fancy 'COVID-19 CHATBOT' when program starts"""
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        init(strip=not sys.stdout.isatty())
        cprint(
            figlet_format(
                text
            ),
            color='yellow',
            attrs=['bold', 'blink']
        )
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def get_letter_at_random_interval(self, answer):
        """Retrieves letter from answer at random interval"""
        answer = re.sub(r'\n+|\r+', '\n', answer).strip()
        answer = '\n'.join([ll.rstrip() for ll in answer.splitlines() if ll.strip()])

        for letter in answer:
            # Adding below random time sleep to give an illusion that it is thinking
            rand_num = random.randrange(1, 100, 1)

            if (rand_num > 98):
                time.sleep(random.randrange(1, 1000, 1) / 1000.0)
            elif (rand_num > 96):
                time.sleep(random.randrange(1, 500, 1) / 1000.0)
            else:
                time.sleep(random.randrange(1, 3, 1) / 1000.0)

            yield letter


###########################################################################################

# Creating COVID Chatbot GUI with tkinter
import random
from tkinter import *


def main():
    """main function that starts the covid-19 information chatbot"""
    log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
    ml = PythonFAQChatbot()
    ml.print_chatbot_ready_text('PYTHON CHATBOT BY AMITESH')
    welcome_text = 'Hi, I am PYTHON bot. You can ask me any questions related to Python!!!'

    def send():
        """Sends text to the text box widget for user question and answer for the user question"""
        user_question = EntryBox.get("1.0", 'end-1c').strip()
        EntryBox.delete("0.0", END)
        ChatLog.config(state=NORMAL)
        if (user_question != ''):
            ChatLog.insert(END, user_question + '\n\n', 'you_text')
            ChatLog.update()

            ChatLog.insert(END, "Bot: ", 'bot')
            ChatLog.update()

            # Get answer for the user question
            answer = ml.get_answer_for_most_similar_title_plus_body(user_question)

            for letter in ml.get_letter_at_random_interval(answer):
                ChatLog.insert(END, letter, 'bot_text')
                ChatLog.update()
                ChatLog.yview(END)

            ChatLog.insert(END, '\n\n', 'bot_text')
            ChatLog.insert(END, "You: ", 'you')
            ChatLog.update()
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)

    base = Tk()
    base.title("COVID-19 Information Bot")
    base.geometry("1100x700")
    base.resizable(width=FALSE, height=FALSE)

    # Create Chat window
    ChatLog = Text(base, bd=0, bg="black", height="8", width="50", font="Arial", )
    ChatLog.config(state=DISABLED)
    ChatLog.tag_config('you', foreground="#ffa500", font=("Ariel", 14, "bold"))
    ChatLog.tag_config('bot', foreground="#7cec12", font=("Ariel", 14, "bold"))
    ChatLog.tag_config('you_text', foreground="#ffa500", font=("Verdana", 13))
    ChatLog.tag_config('bot_text', foreground="#7cec12", font=("Verdana", 13))

    # Bind scrollbar to Chat window
    scrollbar = Scrollbar(base, command=ChatLog.yview)
    ChatLog['yscrollcommand'] = scrollbar.set

    # Create Button to send message
    SendButton = Button(
        base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=5, bd=0,
        highlightbackground="#32de97",
        highlightcolor="#008000", fg='#000000', command=send
    )

    # Create the box to enter message
    EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial",
                    selectborderwidth=2)

    # Place all components on the screen
    scrollbar.place(x=1076, y=6, height=586)
    ChatLog.place(x=6, y=6, height=586, width=1070)
    SendButton.place(x=6, y=601, height=90)
    EntryBox.place(x=128, y=601, height=90, width=965)

    EntryBox.focus_set()

    # Insert welcome text
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "Bot: ", 'bot')
    ChatLog.insert(END, welcome_text + '\n\n', 'bot_text')
    ChatLog.insert(END, "You: ", 'you')
    ChatLog.config(state=DISABLED)
    ChatLog.update()

    base.mainloop()

if __name__ == '__main__':
    try:

        main()

    except KeyboardInterrupt:
        log.critical('Keyboard Interrupted!!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
