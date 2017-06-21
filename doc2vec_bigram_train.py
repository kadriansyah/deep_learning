import nltk
import gensim
from os import listdir
from os.path import isfile, join
from nltk.tokenize import RegexpTokenizer
from gensim.models.phrases import Phraser

# loading bigram model
print('loading bigram model... ')
bigram = Phraser.load('bigram.model')

# create training corpus
print('create training corpus... ')
LabeledSentence = gensim.models.doc2vec.LabeledSentence
docLabels = [f for f in listdir('documents') if f.endswith('.txt')]
data = []
for doc in docLabels:
    print('processing... '+ doc)
    content = open('documents/' + doc, 'r').read().split('\n')
    lines = [line for line in content if line != '']
    content = ' '.join(lines)

    # clean punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [token for token in bigram[tokenizer.tokenize(content.lower())]]
    content = ' '.join(tokens)
    data.append(content)

class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.doc_list = doc_list
        self.labels_list = labels_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            print(self.labels_list[idx].replace('.txt',''))
            yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx].replace('.txt','')])

train_item = LabeledLineSentence(data, docLabels)
# use fixed learning rate
model = gensim.models.Doc2Vec(size=300, window=10, min_count=5, workers=5, alpha=0.025, min_alpha=0.025, iter=10)
model.build_vocab(train_item)
model.train(train_item, total_examples=model.corpus_count, epochs=model.iter)
model.save('doc2vec.model')
