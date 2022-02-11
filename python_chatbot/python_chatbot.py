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
import torch
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
from transformers import (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer)
from tqdm.notebook import tqdm

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

        self.set_dafa_frame_from_csv(QUESTIONS_DATA)
        # setup df questions
        # self.setup_df()

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

        self.clean_column('Title')

        # Load model
        config_class, model_class, tokenizer_class = GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
        model = model_class.from_pretrained('gpt2')
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        preprocessedTitle = self.df['preprocessedTitle'].values
        QID = self.df['Id'].values

        encodedpreprocessedTitle = self.tokenizer.batch_encode_plus(preprocessedTitle)['input_ids']
        self.embeddigs = model.transformer.wte
        print("Shape of embedding matrix : ", self.embeddigs.weight.shape)
        print("Type of embedding matrix : ", type(self.embeddigs))

        TitleEmbeddingList = []
        self.QIDList = []
        for idx, (qid, encodedTitle) in tqdm(enumerate(zip(QID, encodedpreprocessedTitle))):
            if len(encodedTitle) > 0:
                embeddedTitle = self.embeddigs(torch.tensor(encodedTitle).to(torch.int64)).mean(axis=0)
                TitleEmbeddingList.append(embeddedTitle)
                self.QIDList.append(qid)

        numQ = len(TitleEmbeddingList)
        embedDim = len(TitleEmbeddingList[0])
        print("Number of Titles : ", numQ, " and Length of vector of each Title : ", embedDim)
        print("Type of TitleEmbeddingList : ", type(TitleEmbeddingList))

        self.TitleEmbeddingTensor = torch.cat(TitleEmbeddingList, dim=0)
        self.TitleEmbeddingTensor = torch.reshape(self.TitleEmbeddingTensor, (numQ, embedDim))
        print("Shape of TitleEmbeddingTensor : ", self.TitleEmbeddingTensor.shape)
        print("Type of TitleEmbeddingTensor : ", type(self.TitleEmbeddingTensor))

        self.getMostSimilarQuestions(5, "How to MUltiply 2 columns pandas ?", self.df, self.QIDList)

        # log.debug(f"{self.df[['Title', 'Body', 'clean_text']]}")

        self.nlp_for_user_question = spacy.load('en_core_web_lg')

        # Sample question (to create similarity column) - > next time onwards it gets fast
        # self.get_answer_for_most_similar_title_plus_body('How to add two lists')
        log.debug(f'Leaving: "{inspect.currentframe().f_code.co_name}"')

    def preprocesstext(self, text):
        text = " ".join([word for word in text.split(" ") if word not in stop_words])
        text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
        text = text.lower()
        return text

    def getMostSimilarQuestionsIdx(self, K, a, b):
        a_norm = a / a.norm(dim=1)[:, None]
        b_norm = b / b.norm(dim=1)[:, None]
        res = torch.mm(a_norm, b_norm.transpose(0, 1)).squeeze(0)
        res = res.tolist()
        mostSimIdx = sorted(range(len(res)), key=lambda x: res[x])[-K:]
        return mostSimIdx

    def getMostSimilarQuestions(self, K, input, QuestionDF, QIDList):
        input = input
        preprocessedinput = self.preprocesstext(input)
        inputEncoded = self.tokenizer.batch_encode_plus([preprocessedinput])['input_ids']
        inputEmbedded = self.embeddigs(torch.tensor(inputEncoded).to(torch.int64)).squeeze(0).mean(
            axis=0).unsqueeze(0)
        mostSimilarIdx = self.getMostSimilarQuestionsIdx(K, inputEmbedded, self.TitleEmbeddingTensor)
        mostSimilarIdx.reverse()
        print("Most similar ", K, " questions : ")
        parent_ids = []
        for idx, simidx in enumerate(mostSimilarIdx):
            IDQ = QuestionDF[QuestionDF['Id'] == QIDList[simidx]][['Id', 'Title']].values
            parentId = IDQ[0][0]
            simQuestion = IDQ[0][1]
            print((idx + 1), "Question Id : ", parentId, "Question : ", simQuestion)
            parent_ids.append(parentId)

        return parent_ids

    def set_dafa_frame_from_csv(self, csv):
        """Sets data frame based on the csvs passed

        :param string csv: path of csv file

        :returns pd.dataFrame: sets data frame based on the csv passed

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')

        self.df = pd.read_csv(
            csv,
            encoding="ISO-8859-1"
        )
        # self.df = self.df[
        #     self.df['Id'].isin(
        #         list(self.df_tags[self.df_tags['Tag'].isin(['Python', 'python'])]['Id'])
        #     )
        # ]
        # self.df = self.df.sort_values(by=['Id'])
        # self.df.dropna(inplace=True)
        # self.df = self.df.reset_index(drop=True)

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

    def clean_column(self, col):
        """Clean the sentence

        :param string text: text to be cleaned
        :param boolean lemmatize: lemmatize sentence if True else Don't

        :returns string: clean sentence

        """
        self.df['preprocessed' + col] = self.df[col].str.lower()
        self.df['preprocessed' + col] = self.df['preprocessed' + col].str.replace(r'\([^)]*\)', '')
        self.df['preprocessed' + col] = self.df['preprocessed' + col].str.replace('"', '')

        self.df['preprocessed' + col] = self.df['preprocessed' + col].swifter.apply(
            lambda x: ' '.join([CONTRACTIONS_DICT[t] if t in CONTRACTIONS_DICT else t for t in
                                x.split(" ")]))

        self.df['preprocessed' + col] = self.df['preprocessed' + col].str.replace(r"'s\b", "")
        self.df['preprocessed' + col] = self.df['preprocessed' + col].str.replace('[^a-zA-Z0-9 ]', '')

        # For reducing words to their root form
        lemma = WordNetLemmatizer()

        self.df['preprocessed' + col] = self.df['preprocessed' + col].swifter.apply(lambda x: (" ".join(
            [lemma.lemmatize(word, 'v') for word in word_tokenize(x) if
             (word) not in set(stopwords.words('english'))])).strip())

    def clean_sentence(self, sentence, lemmatize=True):
        """Clean the sentence

        :param string sentence: sentence to be cleaned
        :param boolean lemmatize: lemmatize sentence if True else Don't

        :returns string: clean sentence

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        # Cleaning sentence
        new_string = str(sentence).lower()
        new_string = re.sub(r'\([^)]*\)', '', new_string)
        new_string = re.sub('"', '', new_string)
        new_string = ' '.join([CONTRACTIONS_DICT[t] if t in CONTRACTIONS_DICT else t for t in
                              new_string.split(" ")])
        new_string = re.sub(r"'s\b", "", new_string)
        new_string = re.sub("[^a-zA-Z]", " ", new_string)

        words = word_tokenize(new_string)

        # For reducing words to their root form
        lemma = WordNetLemmatizer()

        # if lemmatize True then lemmatize sentence and remove stopwords
        if (lemmatize):
            stop_words = set(stopwords.words('english'))
            words = [lemma.lemmatize(word, 'v') for word in words if (word) not in stop_words]

        return (" ".join(words)).strip()

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

    def get_answer_for_most_similar_title_plus_body(self, user_question):
        """Checks the dataframe and looks for most similar question in df to user question

        :param string user_question: user question

        :returns string: best suited answer for user's question

        """
        log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
        log.debug('Creating question similarity column')
        most_similar_id = self.getMostSimilarQuestions(
            5, user_question, self.df, self.QIDList
        )[0]


        # Now iterate df based on sorted index values and check for which index we find user
        # question's nouns, verbs and adjective in 'clean_answer' or 'clean_question' column
        try:

            df_answers = self.df_answers[
                self.df_answers['ParentId'] == most_similar_id
            ].sort_values(by='Score', ascending=False)
            top_anwer = df_answers.iloc[0]['Body']
            return BeautifulSoup(top_anwer, 'html.parser').get_text()

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

# Creating Python Chatbot using telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters


def main():
    """Start the bot."""
    log.debug(f'Entering: "{inspect.currentframe().f_code.co_name}"')
    ml = PythonFAQChatbot()
    ml.print_chatbot_ready_text('PYTHON CHATBOT BY AMITESH')
    welcome_text = 'Hi, I am PYTHON bot. You can ask me any questions related to Python!!!'

    # Enable logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    logger = logging.getLogger(__name__)

    # Define a few command handlers. These usually take the two arguments update and
    # context. Error handlers also receive the raised TelegramError object in error.
    def start(update, context):
        """Send a message when the command /start is issued."""
        update.message.reply_text('PYTHON CHATBOT BY AMITESH')
        update.message.reply_text('Hi, I am PYTHON bot. You can ask me any questions related to Python!!!')

    def help(update, context):
        """Send a message when the command /help is issued."""
        update.message.reply_text('Help!')

    def echo(update, context):
        """Echo the user message."""
        # Get answer for the user question
        answer = ml.get_answer_for_most_similar_title_plus_body(
            user_question=ml.clean_sentence(update.message.text)
        )
        update.message.reply_text(answer)

    def error(update, context):
        """Log Errors caused by Updates."""
        logger.warning('Update "%s" caused error "%s"', update, context.error)



    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater('1688360137:AAGO2lg61BUUrVLV0l4D-5fzuSV--FLAUkQ', use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()


if __name__ == '__main__':
    try:

        main()

    except KeyboardInterrupt:
        log.critical('Keyboard Interrupted!!!')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
