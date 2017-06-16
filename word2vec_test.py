from gensim.models import Word2Vec

print('loading word2vec...')
min_count = 2
size = 50
window = 4
model = Word2Vec.load('word2vec_model')
