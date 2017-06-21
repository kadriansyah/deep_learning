import nltk
from os import listdir
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser

docLabels = [f for f in listdir('documents') if f.endswith('.txt')]
sentences = []
for doc in docLabels:
    print('processing... '+ doc)
    lines = open('documents/' + doc, 'r').read().split('\n')
    lines = [line for line in lines if line != '']
    content = ' '.join(lines)

    # sentence = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(content.lower())]

    tokenizer = RegexpTokenizer(r'\w+')
    sentence = [tokenizer.tokenize(sent) for sent in nltk.sent_tokenize(content.lower())]
    sentences.extend(sentence)

phrases = Phrases(sentences)
bigram = Phraser(phrases)
print(list(bigram[sentences]))
bigram.save('bigram.model')
