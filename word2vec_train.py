import nltk
from os import listdir
from nltk.tokenize import RegexpTokenizer
from gensim.models import Word2Vec

# create stemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()

docLabels = [f for f in listdir('documents') if f.endswith('.txt')]
data = []
for doc in docLabels:
    print('processing... '+ doc)
    lines = open('documents/' + doc, 'r').read().split('\n')
    lines = [line for line in lines if line != '']
    content = ' '.join(lines)

    # clean punctuation & make lower case
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [stemmer.stem(token.lower()) for token in tokenizer.tokenize(content)]
    data.append(tokens)
print(data)

# data = open ('corpus.txt', 'r').read().split('\n')
# lines = [line for line in data if line != '']
# data = ' '.join(lines)
#
# sentences = nltk.sent_tokenize(data)
# tokenizer = RegexpTokenizer(r'\w+')
# sentences = [tokenizer.tokenize(sent) for sent in sentences]

# train_data = []
# for sent in sentences:
#     data = [token.lower() for token in sent]
#     train_data.append(data)
# print(train_data)

print('training word2vec...')
min_count, size, window = 2, 100, 5
model = Word2Vec(data, min_count=min_count, size=size, window=window)
model.save('word2vec.model')
for i in range(1, 11):
    print('training word2vec... %d' % model.train_count)
    model.train(data, total_examples=model.corpus_count, epochs=model.iter)
    model.save('word2vec.model')
