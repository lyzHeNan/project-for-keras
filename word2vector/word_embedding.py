# conding:utf-8
# lyz
# 10.10.2017
# 
import gensim
import jieba
import sys,os
import re
import numpy as np


def train_model():
	print('scripts name:',sys.argv[0])
	file_dir = sys.argv[1]
	model_dir = sys.argv[2]
	assert file_dir.endswith('.txt') == True
	sentences = list()
	with open(sys.argv[1],'r') as f:
		for line in f.readlines():
			line = line.strip()
			line = re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+",'',line)
			sentences.append(list(jieba.cut(line)))
	print (sentences[0])
	#sentences = list(jieba.cut(sentences))
	if os.path.exists(model_dir):
		print ('Loading model......')
		model = gensim.models.Word2Vec.load(model_dir)
	else:
		print ('Training model......')
		model = gensim.models.Word2Vec(sentences,size=100,min_count=0,workers=10)
		model.save(model_dir)
	print(model)
	# print all the words
	print (model.wv.vocab.keys())
	return model
def sentenceEm(model,k,sentence):
	sentence = sentence.strip()
	sentence = re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+",'',sentence)
	sen_cut = list(jieba.cut(sentence))
	print (sen_cut)
	num_cut = len(sen_cut)
	sentence_embedding = list()
	for word in sen_cut:
		wv_tmp = model.wv[word][:k]
		sentence_embedding += list(wv_tmp)
	sembedding = [sentence/(float(num_cut)) for sentence in sentence_embedding][:k]
	#sembedding = [sentence for sentence in sentence_embedding][:k]
	print (sembedding)
	return sembedding
	
if __name__ == '__main__':
	model = train_model()
	sentence_01 = '想请教下，江苏靖江市怎么到扬泰机场啊'
	sentence_02 = '想问下，江苏靖江市怎么到扬泰机场啊'
	vec_01 = sentenceEm(model,60,sentence_01)
	vec_02 = sentenceEm(model,60,sentence_02)
	print (type(vec_01))
	print (np.dot(np.array(vec_01),np.array(vec_02)))
	
