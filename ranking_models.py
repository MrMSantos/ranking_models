import json
import spacy
import pickle
import requests
import simplejson
import pytrec_eval
import numpy as np

from scipy.stats import rankdata
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Add, Activation, Lambda

#TODO CHECK OTHER FEATURES (TF, IDF), 2003NP (error with &), VALIDATION SET MAY BE UNBALANCED (0 >> 1)

#Load Language Model
nlp = spacy.load("en_trf_bertbaseuncased_lg")

def bert_tokens(query, document):

	if query[-1] not in ['.', '?']:
		query += '.'
	document = document.replace('\t', '')
	document = document.replace('\n', '')
	sentence = query + ' ' + document

	bert_transf = nlp(sentence)
	first_token = bert_transf._.trf_last_hidden_state[0]
	last_token = bert_transf._.trf_last_hidden_state[-1]
	token = np.concatenate([first_token, last_token])

	return token

def classifier_svc(X, y):

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
	model = LinearSVC(max_iter = 100000)
	model.fit(X_train, y_train)
	
	print(cross_val_score(model, X_train, y_train, cv = 5))
	print(model.score(X_test, y_test))

	return model

def score_model(input_shape):

	model = Sequential()
	model.add(Dense(1024, activation = 'relu', input_dim = input_shape))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(512, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(256, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(128, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(1, activation = 'sigmoid'))

	return model

def rank_model(input_shape):

	model = score_model()

	rel_docs = Input(shape = (input_shape, ))
	irr_docs = Input(shape = (input_shape, ))
	rel_docs_y = model(rel_docs)
	irr_docs_y = model(irr_docs)

	added = Add()([rel_docs_y, irr_docs_y])
	diff = Lambda(lambda x: 1 - x, output_shape = (1, ))(added)
	y_new = Activation('tanh')(diff)

	rank_model = Model(inputs = [rel_docs, irr_docs], outputs = y_new)

	return rank_model

def solr_search(query, docs_returned = 1):

	url = 'http://localhost:8983/solr/gov'
	url += '/query?q={0}'.format(query)
	url += '&fl=id,score,title&rows={0}&q.op=OR'.format(docs_returned)
	response = requests.request('GET', url)
	response_json = simplejson.loads(response.text)

	return response_json

def training_set(docs_returned, topics_date):

	X = []
	#X_rel = []
	#X_irr = []
	y = []

	for TOPICS in topics_date:
	#Load topics dictionary
		with open('topics/topics_' + TOPICS + '.pickle', 'rb') as f:
			topics = pickle.load(f)
			TOPICS_TOTAL = len(topics)
		with open('topics/qrels_' + TOPICS + '.pickle', 'rb') as f:
			qrel = pickle.load(f)
		print('YEAR:', TOPICS)
		#Feature Extraction
		for query in topics:
			response_json = solr_search(topics[query], docs_returned)
			query_text = topics[query]

			for (rank, document) in enumerate(response_json['response']['docs']):
				doc = document['id'][-19:-5]
				
				#Feature Extraction BERT
				if 'title' in document:
					title_text = document['title'][0]
				else:
					title_text = ''
				
				token = bert_tokens(query_text, title_text)
				X.append(token)
				if doc in qrel[query]:
					y.append(qrel[query][doc])
				else:
					y.append(0)
				'''
				if doc in qrel[query] and qrel[query][doc] == 1:
					token = bert_tokens(query_text, title_text)
					X_rel.append(token)
					y.append(1)
				elif len(X_rel) > len(X_irr):
					token = bert_tokens(query_text, title_text)
					X_irr.append(token)'''

	X = np.asarray(X)
	#X_rel = np.asarray(X_rel)
	#X_irr = np.asarray(X_irr)
	y = np.asarray(y)

	return X, y

def test_set(docs_returned, topics_date, rerank = 10, model = None):

	with open('topics/topics_' + topics_date + '.pickle', 'rb') as f:
		topics = pickle.load(f)
	with open('topics/qrels_' + topics_date + '.pickle', 'rb') as f:
		qrel = pickle.load(f)

	run = {}
	for query in topics:
		run[query] = {}
		json = solr_search(topics[query], docs_returned)
		
		for (rank, document) in enumerate(json['response']['docs']):
			doc = document['id'][-19:-5]
			score = document['score']
			run[query][doc] = float(score)

	for query in run:
		top_docs = sorted(run[query], key = run[query].get, reverse = True)[:rerank]
		query_text = topics[query]

		for doc in top_docs:
			doc_path = 'id:\"/mnt/c/Users/Santos/Desktop/GOV/scripts/corpus/{0}.html\"'.format(doc)
			solr_doc = solr_search(doc_path)
			document = solr_doc['response']['docs'][0]
			solr_score = document['score']

			if 'title' in document:
				title_text = document['title'][0]
			else:
				title_text = ''

			token = [bert_tokens(query_text, title_text)]
			X_test = np.asarray(token)
			model_score = model.predict(X_test)[0][0]

			prediction = rr_fusion([solr_score, model_score])
			run[query][doc] = prediction * 1000

	evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map', 'Rprec', 'P', 'ndcg'})
	eval_dict = evaluator.evaluate(run)

	return eval_dict

def rr_fusion(scores, k = 60):

	rrf_score = 0
	for score in scores:
		rrf_score += 1 / (score + k)

	return rrf_score

def results(eval_dict):

	mapList = []
	ndcgList = []
	map_total, ndcg_total, rprec_total, p10_total = 0, 0, 0, 0
	total_queries = len(eval_dict) 

	for query in eval_dict:
		value_map = eval_dict[query]['map']
		map_total += value_map

		value_ndcg = eval_dict[query]['ndcg']
		ndcg_total += value_ndcg

		value_rprec = eval_dict[query]['Rprec']
		rprec_total += value_rprec

		value_p10 = eval_dict[query]['P_10']
		p10_total += value_p10

	print('MAP:\t', round(map_total / total_queries, 4))
	print('NDCG:\t', round(ndcg_total / total_queries, 4))
	print('R-P:\t', round(rprec_total / total_queries, 4))
	print('P@10:\t', round(p10_total / total_queries, 4))




if __name__ == '__main__':
	
	docs_returned_train = 100
	docs_returned_test = 10000

	topics_train = ['2003', '2004']
	topics_test = '2002'

	X, y = training_set(docs_returned_train, topics_train)
	input_shape = X.shape[1]

	BATCH_SIZE = 128
	EPOCHS = 30

	score_model = score_model(input_shape)
	score_model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
	score_model.fit(X, y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)

	#rank_model = rank_model(input_shape)
	#rank_model.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])
	#rank_model.fit([X_rel, X_irr], y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)

	eval_dict = test_set(docs_returned_test, topics_test, rerank = 20, model = score_model)
	results(eval_dict)
