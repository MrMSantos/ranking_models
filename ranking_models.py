import json
import spacy
import pickle
import requests
import simplejson
import numpy as np

from scipy.stats import rankdata
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Input, Add, Subtract, Activation, Lambda

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

def score_model():

	model = Sequential()
	model.add(Dense(1024, activation = 'relu'))
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
	model.add(Dense(1, activation = 'tanh'))

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

def import_model(model):

	weights = model.get_weights()
	solr_model = {'name' : 'my_ranknet_model',
				'class' : 'org.apache.solr.ltr.model.NeuralNetworkModel',
				'features' : [
					{ 'name' : 'original_score' },
					{ 'name' : 'max_sim' },
				],
				'params': {}}
	layers = []
	layers.append({'matrix': weights[0].T.tolist(),
				'bias': weights[1].tolist(),
				'activation': 'relu'})
	layers.append({'matrix': weights[2].T.tolist(),
				'bias': weights[3].tolist(),
				'activation': 'relu'})
	layers.append({'matrix': weights[4].T.tolist(),
				'bias': weights[5].tolist(),
				'activation': 'relu'})
	layers.append({'matrix': weights[6].T.tolist(),
				'bias': weights[7].tolist(),
				'activation': 'identity'})
	solr_model['params']['layers'] = layers

	with open('my_ranknet_model.json', 'w') as out:
		json.dump(solr_model, out, indent = 4)

def training_set(docs_returned, topics_date):

	#X = []
	X_rel = []
	X_irr = []
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
			url = 'http://localhost:8983/solr/gov'
			url += '/query?q={0}'.format(topics[query])
			url += '&rq={{!ltr model=my_model efi.text=\"{0}\"}}'.format(topics[query])
			url += '&fl=id,score,title,[features]&rows={0}&q.op=OR'.format(docs_returned)
			response = requests.request('GET', url)
			response_json = simplejson.loads(response.text)

			for (rank, document) in enumerate(response_json['response']['docs']):
				doc = document['id'][-19:-5]
				query_text = topics[query]

				#Feature Extraction BERT
				if 'title' in document:
					title_text = document['title'][0]
				else:
					title_text = ''
				
				'''
				token = bert_tokens(query_text, title_text)
				X.append(token)
				if doc in qrel[query]:
					y.append(qrel[query][doc])
				else:
					y.append(0)'''
				
				if doc in qrel[query] and qrel[query][doc] == 1:
					token = bert_tokens(query_text, title_text)
					X_rel.append(token)
					y.append(1)
				elif len(X_rel) > len(X_irr):
					token = bert_tokens(query_text, title_text)
					X_irr.append(token)

	#X = np.asarray(X_rel)
	X_rel = np.asarray(X_rel)
	X_irr = np.asarray(X_irr)
	y = np.asarray(y)

	return X_rel, X_irr, y



if __name__ == '__main__':
    
	DOCS_RETURNED = 50
	topics = ['2002', '2003', '2004']

	X_rel, X_irr, y = training_set(DOCS_RETURNED, topics)
	input_shape = X_rel.shape[1]

	BATCH_SIZE = 128
	EPOCHS = 30

	#Score Model
	#score_model = score_model()
	#score_model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
	#score_model.fit(X, y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)

	#Rank Model
	rank_model = rank_model(input_shape)
	rank_model.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])
	rank_model.fit([X_rel, X_irr], y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_split = 0.1)
