import gensim

model = gensim.models.Doc2Vec.load('doc2vec.model')
docvecs = model.docvecs
