from gensim.models import Word2Vec

print('loading word2vec...')
model = Word2Vec.load('word2vec.model')
model.init_sims()

def get_related_term(token, topn=10):
    for word, similarity in model.most_similar(positive=[token], topn=topn):
        print('%s %.5f%%' % (word, similarity))

def word_algebra(add=[], substract=[], topn=1):
    answers = model.most_similar(positive=add, negative=substract, topn=topn)
    for term, similarity in answers:
        print(term)
