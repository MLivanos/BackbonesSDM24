'''
Identification and Uses of Deep Learning Backbones via Pattern Mining
Michael Livanos, Ian Davidson
University of California, Davis
SDM 2024
'''

# USAGE
# python recognize.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle --image images/adrian.jpg

# import the necessary packages
import numpy as np
import re
import imutils
from imutils import paths
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
import csv
import pickle
import dlib
import cv2
import os

# Use explanation to predict class
def exp_prediction(exp, neurons, weights, architecture, threshold):
	# Get transactions

	layers = []
	offset = 0
	for layer in architecture:
		layers.append(neurons[offset:layer + offset])
		offset += layer

	activations = []

	j = 0
	indices = []
	for i in range(len(neurons)):
		neurons[i] *= weights[i]
		indices.append(i)
	neurons, indices = (list(t) for t in zip(*sorted(zip(neurons, indices), reverse = True)))
	activations = indices[0 : threshold]

	exp_weights = {}
	weight_total = {}
	ans_weight = {}

	# Generate weights dict, key class name, value dict explanation

	for c in exp.keys():
		explanation = exp[c]
		exp_weights[c] = {}
		for neuron in explanation:
			exp_weights[c][neuron[0]] = neuron[1]
		ans_weight[c] = 0.0
		weight_total[c] = 0.0

	# Sum up the weights for each class for normalization

	for c in exp_weights.keys():
		for el in exp_weights[c].keys():
			weight_total[c] += exp_weights[c][el]

	for neuron in activations:
		for c in exp.keys():
			mul = 1
			if c in exp_weights.keys() and neuron in exp_weights[c].keys():
				ans_weight[c] += mul * exp_weights[c][neuron]
				mul -= 0.02

	# Find exp prediction
	m = ''
	m_value = -1
	for c in ans_weight.keys():
		ans_weight[c] = ans_weight[c] / weight_total[c]
		if ans_weight[c] > m_value:
			m = c
			m_value = ans_weight[c]

	return m



nones = 0
for i in range(10):
	j = i + 1
	# Relevant filepaths
	true_test_path = os.path.join('..', 'Activations', 'activation_' + str(j) + '_plus.csv')
	false_test_path = os.path.join('..', 'Activations', 'activation_' + str(j) + '_minus.csv')
	explanation_path = os.path.join('..', 'Explanations', 'explanation' + str(j) + '.txt')
	weights_path = os.path.join('..', 'Weights', 'weights_' + str(j) + '.csv')

	# Load explanation
	explanation_file = open(explanation_path, 'r')
	explanations = explanation_file.readlines()
	valid_pred = eval(explanations[0])
	invalid_pred = eval(explanations[1])
	#model_confusion_exp = eval(explanations[2])

	# Load weights
	with open(weights_path, newline='') as block:
		weights_reader = csv.reader(block, delimiter=',', quotechar='|')
		for row in weights_reader:
			for i in range(len(row)):
				row[i] = float(row[i])
			weights = row

	# Load correctly predicted neurons
	true_neurons = {}
	with open(true_test_path, newline='') as block:
		neuron_values = csv.reader(block, delimiter=' ',
			quotechar='|')
		for row in neuron_values:
			# Convert strings to floats
			for i in range(len(row) - 1):
				row[i + 1] = float(row[i + 1])
			#row = [float(i) for i in row[0].split(',')]
			# Get all classes ready for evaluation
			if row[0] not in true_neurons.keys():
				true_neurons[row[0]] = []
			# Get each subset of neurons for each class
			true_neurons[row[0]].append(row[1:])

	# Load incorrectly predicted neurons
	false_neurons = {}
	with open(false_test_path, newline='') as block:
		neuron_values = csv.reader(block, delimiter=' ',
			quotechar='|')
		for row in neuron_values:
			# Convert strings to floats
			for i in range(len(row) - 1):
				row[i + 1] = float(row[i + 1])
			#row = [float(i) for i in row[0].split(',')]
			# Get all classes ready for evaluation

			if row[0] not in false_neurons.keys():
				false_neurons[row[0]] = []
			# Get each subset of neurons for each class
			false_neurons[row[0]].append(row[1:])
	'''
	0 - model predicted correctly and exp agreed,
	1 - model predicted correctly, exp disagreed, false inconclusive
	2 - model predicted correctly, exp disagreed, false incorrectly identified
	3 - model predicted incorrectly, exp agreed
	4 - model predicted incorrectly, exp disagreed, false inconclusive
	5 - model predicted incorrectly, exp disagreed, false identified, produced correct adjustment
	6 - model predicted incorrectly, exp disagreed, false identified, produced incorrect adjustment
	[1231, 72, 1248, 169, 80, 0]
	'''
	t = 0
	f = 0

	outcomes = [0, 0, 0, 0, 0, 0, 0]
	outcomes2 = [0, 0, 0, 0, 0, 0]
	architecture = [256, 32]
	for c in true_neurons.keys():
		for activations in true_neurons[c]:
			exp_pred = exp_prediction(invalid_pred, activations, weights, architecture, 50)
			exp_pred2 = exp_prediction(valid_pred, activations, weights, architecture, 50)
			if exp_pred2 == c:
				outcomes2[0] += 1
				t += 1
			else:
				f += 1
			if exp_pred == c and exp_pred2 != c:
				outcomes2[1] += 1
			else:
				outcomes2[2] += 1
				'''invalid_exp_pred = exp_prediction(invalid_pred, activations, weights, [80, 60, 40, 30, 20], 50)
				if invalid_exp_pred != c:
					outcomes[1] += 1
				elif invalid_exp_pred == c:
					outcomes[2] += 1'''

	for c in false_neurons.keys():
		c = -1 * (int(c) - 1)
		for activations in false_neurons[str(c)]:
			exp_pred = exp_prediction(invalid_pred, activations, weights, architecture, 50)
			exp_pred2 = exp_prediction(valid_pred, activations, weights, architecture, 50)
			if exp_pred2 == c:
				t += 1
			else:
				f += 1
			if int(exp_pred) == c and int(exp_pred2) != c:
				outcomes2[3] += 1
			else:
				outcomes2[4] += 1
			if exp_pred2 == c:
				outcomes2[5] += 1
				'''
				invalid_exp_pred = exp_prediction(invalid_pred, activations, weights, [80, 60, 40, 30, 20], 50)
				if invalid_exp_pred != predicted_name:
					outcomes[4] += 1
				else:
					new_prediction = exp_prediction(model_confusion_exp[predicted_name], activations, weights, [80, 60, 40, 30, 20], 50)
					if new_prediction == true_name:
						outcomes[5] += 1
					else:
						outcomes[6] += 1'''

	print (t/(t+f))







	