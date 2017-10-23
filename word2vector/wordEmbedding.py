# conding:utf-8
# lyz
# 10.10.2017
# 
import gensim
import jieba
import sys,os

class sentenceSet(object):
	
	def __init__(self,dirname):
		self.dirname = dirname
	
	def __iter__(self):
		for line in open(self.dirname,'r'):
			yield list(jieba.cut(line))


def train_model():
	print('scripts name:',sys.argv[0])
	file_dir = sys.argv[1]
	model_dir = sys.argv[2]
	assert file_dir.endswith('.txt') == True
	sentences = sentenceSet(file_dir)
	#print (sentences)
	#sentences = list(jieba.cut(sentences))
	print (sentences)
	if os.path.exists(model_dir):
		print ('Loading model......')
		model = gensim.models.Word2Vec.load(model_dir)
	else:
		print ('Training model......')
		model = gensim.models.Word2Vec(sentences,size=100,min_count=10,workers=10)
		model.save(model_dir)
	print(model)
	print (model.wv.vocab.keys())
	for word in (model.wv.vocab):
		print (word)
		result_wv = model.wv[word][:6]
		print (result_wv,result_wv.size)
	return model
def sentenceEm(model,k):
	sentence = '想请教下，江苏靖江市怎么到扬泰机场啊'
	sen_cut = list(jieba.cut(sentence))
	word_num = len(sen_cut)
	sentence_embedding = list()
	for word in sen_cut:
		wv_tmp = model.wv[word][:k]
		sentence_embedding += list(wv_tmp)
	print ([sentence/(float(word_num)) for sentence in sentence_embedding])
	
if __name__ == '__main__':
	model = train_model()
	sentenceEm(model,6)
