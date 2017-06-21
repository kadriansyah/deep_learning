from os import listdir
from os.path import isfile, join
from nltk.tokenize import RegexpTokenizer
import gensim
LabeledSentence = gensim.models.doc2vec.LabeledSentence

docLabels = [f for f in listdir('documents_test') if f.endswith('.txt')]

data = []
for doc in docLabels:
    print('processing... '+ doc)
    content = open('documents_test/' + doc, 'r').read().split('\n')
    lines = [line for line in content if line != '']
    content = ' '.join(lines)

    # clean punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    data.extend([token.lower() for token in tokenizer.tokenize(content)])

model = gensim.models.Doc2Vec.load('doc2vec.model')
inferred_vector = model.infer_vector(data)

sims = model.docvecs.most_similar([inferred_vector], topn=10)
print(sims)
