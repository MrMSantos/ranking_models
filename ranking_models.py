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
from keras.layers import Dense, Dropout, BatchNormalization, Input, Subtract, Activation

#TODO CHECK OTHER FEATURES (TF, IDF), POINT-WISE OR PAIR-WISE, 2003NP (error with &)

#Load Language Model
nlp = spacy.load("en_trf_bertbaseuncased_lg")

DOCS_RETURNED = 50
DATASET = 'gov'
TOPICS_DATE = ['2002', '2003', '2004']

print('RETURNING', DOCS_RETURNED, 'DOCUMENTS')

X = []
X_rel = []
X_irr = []
y = []

for TOPICS in TOPICS_DATE:
#Load topics dictionary
	with open('topics/topics_' + TOPICS + '.pickle', 'rb') as f:
		topics = pickle.load(f)
		TOPICS_TOTAL = len(topics)
	with open('topics/qrels_' + TOPICS + '.pickle', 'rb') as f:
		qrel = pickle.load(f)
	print('YEAR:', TOPICS)
	#Feature Extraction
	for query in topics:
		url = 'http://localhost:8983/solr/' + DATASET
		url += '/query?q={0}'.format(topics[query])
		url += '&rq={{!ltr model=my_model efi.text=\"{0}\"}}'.format(topics[query])
		url += '&fl=id,score,title,[features]&rows={0}&q.op=OR'.format(DOCS_RETURNED)
		response = requests.request('GET', url)

		try:
			response_json = simplejson.loads(response.text)
		except simplejson.JSONDecodeError:
			print('ERROR JSON')
		if "error" in response_json:
			print('ERROR JSON')

		for (rank, document) in enumerate(response_json['response']['docs']):
			doc = document['id'][-19:-5]

			#Feature Extraction BERT
			if 'title' in document:
				title_text = document['title'][0]
			else:
				title_text = ''
			query_text = topics[query]
			
			if query_text[-1] not in ['.', '?']:
				query_text += '.'
			title_text = title_text.replace('\t', '')
			title_text = title_text.replace('\n', '')
			pair_query_doc = query_text + ' ' + title_text
			
			'''token = bert_tokens(pair_query_doc)
			X.append(token)
			if doc in qrel[query]:
				y.append(qrel[query][doc])
			else:
				y.append(0)'''
			
			if doc in qrel[query] and qrel[query][doc] == 1:
				token = bert_tokens(pair_query_doc)
				X_rel.append(token)
				y.append(1)
			elif len(X_rel) > len(X_irr):
				token = bert_tokens(pair_query_doc)
				X_irr.append(token)


def bert_tokens(sentence):
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

def score_model(X, y, batch_size = 128, epochs = 30):

	X = np.asarray(X)
	y = np.asarray(y)

	model = Sequential()
	model.add(Dense(128, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(64, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(32, activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.1))
	model.add(Dense(1, activation = 'linear'))

	model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
	history = model.fit(X, y, batch_size = batch_size, epochs = epochs, validation_split = 0.1)

	return model

def rank_model(X_rel, X_irr, y, batch_size = 128, epochs = 30):

	X_rel = np.asarray(X_rel)
	X_irr = np.asarray(X_irr)
	y = np.asarray(y)
	input_shape = X_rel.shape[1]

	score_model = Sequential()
	score_model.add(Dense(1024, activation = 'relu'))
	score_model.add(BatchNormalization())
	score_model.add(Dropout(0.1))
	score_model.add(Dense(512, activation = 'relu'))
	score_model.add(BatchNormalization())
	score_model.add(Dropout(0.1))
	score_model.add(Dense(256, activation = 'relu'))
	score_model.add(BatchNormalization())
	score_model.add(Dropout(0.1))
	score_model.add(Dense(128, activation = 'relu'))
	score_model.add(BatchNormalization())
	score_model.add(Dropout(0.1))
	score_model.add(Dense(1, activation = 'linear'))

	rel_docs = Input(shape = (input_shape, ))
	irr_docs = Input(shape = (input_shape, ))
	rel_docs_y = score_model(rel_docs)
	irr_docs_y = score_model(irr_docs)

	diff = Subtract()([rel_docs_y, irr_docs_y])

	y_new = Activation('sigmoid')(diff)

	rank_model = Model(inputs = [rel_docs, irr_docs], outputs = y_new)
	rank_model.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])
	history = rank_model.fit([X_rel, X_irr], y, batch_size = batch_size, epochs = epochs, validation_split = 0.1)

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




#model = score_model(X, y)
model = rank_model(X_rel, X_irr, y)
#import_model(model)
