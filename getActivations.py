'''
Identification and Uses of Deep Learning Backbones via Pattern Mining
Michael Livanos, Ian Davidson
University of California, Davis
SDM 2024
'''

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, LeakyReLU, MaxPooling2D, Flatten
from scipy.special import softmax
import numpy as np
import os, csv

#14, 16
def get_activations(model, layers, X):
	outputs = []
	for layer in layers:
		outputs.append(model.layers[layer].output)
	output_function = K.function([model.layers[0].input], outputs)
	output = output_function([np.stack(X).reshape(-1, 1000, 80, 1)])
	activations = []
	for layer in output:
		activations += layer.tolist()
	if activations[-1][0] <= 0.5:
		activations[-1][0] = 0
	else:
		activations[-1][0] = 1
	return activations

def get_row(correct, activations):
	row = ''
	if correct:
		row += str(activations[-1][0]) + ' '
	else:
		row += str(-1 * (activations[-1][0] - 1)) + ' '
	for layer in activations:
		if len(layer) == 1:
			break
		for activation in layer:
			row += str(activation) + ' '
	row = row[0:-1]
	row += '\n'
	return row

for j_prime in range(10):
	j = j_prime + 1
	# Load and process labels
	print ('Processing labels')
	training_labels = []
	instance_order = []
	training_label_path = os.path.join('..', 'Labels', 'bird_' + str(j) + '.csv')
	with open(training_label_path, newline='') as csvfile:
		line_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in line_reader:
			training_labels.append(int(row[1]))
			instance_order.append(row[0])

	print ('Processing spectrograms')
	training_instances = []
	validation_instances = []
	training_files = {}
	training_instances_path = os.path.join('..', 'Data', 'F' + str(j) + '_train')
	training_files_list = os.listdir(training_instances_path)
	for f in training_files_list:
		hash_key = f.split('.')[0]
		training_files[hash_key] = os.path.join(training_instances_path, f)
	for hash_key in instance_order:
		path = training_files[hash_key]
		# Ignore irrelevant files, namely DS store on Mac OS
		if path.split('.')[-1] == 'npz':
			training_instances.append(np.load(path, allow_pickle=True)['melspect2048'])

	print ('Normalizing spectrograms')
	# Format input
	index = 0
	for input_arr in training_instances:
		if input_arr.shape[0] < 1000:
			left = 1000 - input_arr.shape[0]
			z = np.zeros((left, 80))
			input_arr = np.append(input_arr, z)
			input_arr = input_arr.reshape(1000, 80)
		else:
			input_arr = input_arr[:1000]
		training_instances[index] = input_arr
		index += 1

	model = Sequential([
			Conv2D(16, 3, input_shape=(1000, 80, 1), data_format='channels_last'),
			LeakyReLU(alpha=0.01),
			MaxPooling2D(pool_size=3),
			Conv2D(16, 3),
			LeakyReLU(alpha=0.01),
			MaxPooling2D(pool_size=3),
			Conv2D(16, (3, 1)),
			LeakyReLU(alpha=0.01),
			MaxPooling2D(pool_size=(3, 1)),
			Conv2D(16, (3, 1)),
			LeakyReLU(alpha=0.01),
			MaxPooling2D(pool_size=(3, 1)),
			Flatten(),
			Dense(256),
			LeakyReLU(alpha=0.01),
			Dense(32),
			LeakyReLU(alpha=0.01),
			Dense(1, activation='sigmoid')])
	model_path = os.path.join('..', 'Models', 'Networks', 'bulbul' + str(j) + '.h5')
	model.load_weights(model_path)
	print ("Loaded model from disk")

	correct = 0
	incorect = 0
	i = 0
	plus_seq = []
	minus_seq = []
	for instance in training_instances:
		activations = get_activations(model, [14, 16, 17], instance)
		flattened = activations[0] + activations[1]
		if (activations[-1][0] == training_labels[i]):
			correct += 1
			row = get_row(True, activations)
			plus_seq.append(row)
		else:
			incorect += 1
			row = get_row(False, activations)
			minus_seq.append(row)
		i += 1
	with open(os.path.join('..', 'Activations', 'activation_' + str(j) + '_plus.csv'), 'w') as f:
		f.writelines(plus_seq)
	with open(os.path.join('..', 'Activations', 'activation_' + str(j) + '_minus.csv'), 'w') as f:
		f.writelines(minus_seq)
	print (correct)
	print (incorect)

	#activations = get_activations(model, [14, 16, 17], training_instances[0])
	#print (activations)














