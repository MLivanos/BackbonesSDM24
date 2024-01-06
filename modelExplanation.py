'''
Identification and Uses of Deep Learning Backbones via Pattern Mining
Michael Livanos, Ian Davidson
University of California, Davis
SDM 2024
'''

from orangecontrib.associate.fpgrowth import *
from mlxtend.frequent_patterns import fpmax, fpgrowth
import pandas as pd
import csv, os, time, numpy

''' Pipeline: Find activations -> transaction database ->
find maxm_insup -> treshold minsup -> mine explanation '''

class ClassExplanation():
	def __init__(self, label, neurons, test_neurons, weights, architecture,
		threshold, minsup_threshold = 0.0, algorithm = 'fpmax'):

		'''Arguments:
		Label: metadata

		neurons (list of float/int): activations from linearized
		network output

		test_neurons (list of float/int): segmented out of a lartger test
		TODO: Do this automatically

		architecture (list of int): directly imported from
		ModelExplanation

		threshold (int): directly imported from ModelExplanation

		minsup_threshold (float): directly imported from
		ModelExplanation

		algorithm (string): algorithm for frequent itemset mining.
		Currently supported algorithm(s) is/are:
		FP-Growth (fpgrowth)'''

		self.label = label
		self.architecture = architecture
		self.threshold = threshold
		self.minsup_threshold = minsup_threshold
		self.algorithm = algorithm
		self.original_size = len(neurons)
		self.neurons = neurons
		self.test_neurons = test_neurons
		self.total_neurons = sum(architecture)
		self.weights = weights

		'''Find which neurons are considered "activated" based on the
		threshold, create binary  transaction log based off of those
		activations'''
		self.test_activations = self.find_transactions(self.test_neurons)

		sparse = False
		if self.algorithm == 'fpmax':
			sparse = True

		self.transactions = self.find_transactions(self.neurons, sparse=sparse)

	''' Find the k most activated nodes, returns list of strings in the
	format layer_neuron, where top neurons are considered index 0 and
	the first layer is considered 0 indicating what will be
	considered activated'''
	def find_activation(self, neurons, offset):
		''' Weight activations by sum of weighted weights for each
		neuron '''
		for i in range(len(neurons)):
			neurons[i] = neurons[i] * self.weights[i + offset]
		activations = []
		# Create an index list to act as the name of the neuron
		indicies = []
		threshold = int(self.threshold * len(neurons) / self.total_neurons)
		for i in range(len(neurons)):
			indicies.append(i)
		# Sort neuron and index lists in parallel
		neurons, indicies = (list(t) for t in zip(*sorted(zip(neurons,
			indicies), reverse = True)))
		for i in range(threshold):
			activations.append(indicies[i] + offset)
		return activations

	'''Finds the associated layer given an index of the neuron list'''
	def find_layer(self, index):
		count = 0
		for i in range(len(self.architecture)):
			if index > count + self.architecture[i]:
				count += self.architecture[i]
			else:
				return i

	'''Creates a dictionary to map a given neuron to its corresponding
	layer in O(1) time'''
	def create_layer_dict(self):
		self.layer_dict = {}
		for i in range(sum(self.architecture)):
			layer_dict[i] = self.find_layer(i)

	'''Given a set of neurons, a threshold for activation (k), create a
	transaction list'''
	def find_transactions(self, neurons, sparse=False):
		transactions = []
		for row in neurons:
			layers = []
			count = 0
			for l in self.architecture:
				layers.append(row[count: count + l])
				count = count + l
			transaction = []
			i = 0
			for layer in layers:
				transaction += self.find_activation(layer,
					sum(self.architecture[0 : i]))
				i += 1
			transactions.append(transaction)
		# FP Max algorithm takes sparse binary pandas dataframes
		if sparse:
			sparse_transactions = []
			for i in range(len(transactions)):
				sparse_transactions.append([0] * sum(self.architecture))
				for el in transactions[i]:
					sparse_transactions[i][el] = 1
			transactions = pd.DataFrame(sparse_transactions)
		return transactions

	'''Given trasnactions, find the maximally frequent itemsets that
	create a complete graph and associated minsup'''
	def find_max_minsup(self):
		terminal = False
		graphs = []
		minsup = 1.0
		decrement = max(0.01, 1/len(self.transactions))
		while (not terminal):
			if minsup < 0:
				input ()
			#print ("        " + str(minsup))
			# Generate potential graphs via frequent itemset mining
			try:
				potential_graph_tups = fpmax(self.transactions,
					min_support=minsup)
				potential_graph_tups = potential_graph_tups.values.tolist()
			except:
				potential_graph_tups = []
			''' Find each generated subgraph and see if is a complete graph,
			is so, stop'''
			# Potential speedup
			potential_graphs = []
			lengths = []
			for tup in potential_graph_tups:
				g = list(tup[1])
				potential_graphs.append(g)
				lengths.append(len(g))
			if len(lengths) >= 1:
				lengths, potential_graphs = (list(t) for t in zip(*sorted(zip(lengths,
				potential_graphs), reverse = True)))
			i = 0
			for graph in potential_graphs:
				if lengths[i] < len(self.architecture):
					break
			# End potential speedup
				if self.complete_graph(graph):
					#print (graph)
					print (minsup + decrement)
					graphs.append(graph)
					terminal = True
				i += 1
			minsup -= decrement
		return (minsup + decrement)

	'''def cover_based_explanation(self):
		minsup = self.find_max_minsup()
		max_minsup = minsup
		current_size = 1
		explanation = []
		neurons = []
		to_delete = []
		while (self.transactions.shape[0] >= self.cover_threshold * self.original_size):
			potential_graph_tups = fpmax(self.transactions, minsup)
			potential_graph_tups = potential_graph_tups.values.tolist()
			found = False
			for tup in potential_graph_tups:
				graph = list(tup[1])
				support = tup[0]
				if self.complete_graph(graph):
					for i in range(len(graph)):
						if graph[i] not in neurons:
							explanation.append((graph[i], current_size * minsup/max_minsup))
							neurons.append(graph[i])
					found = True
					for el in graph:
						if el not in to_delete:
							to_delete.append(el)
			if found:
				self.search_transactions(graph)
				if len(self.transactions) > 0:
					current_size = len(self.transactions)/self.original_size
					minsup = self.find_max_minsup()
				to_delete = []
		return explanation
'''
	def search_transactions(self, graph, transactions):
		pop_indicies = []
		for index, row in transactions.iterrows():
			covered = True
			for el in graph:
				if transactions.loc[index,el] == 0:
					covered = False
					break
			if covered:
				pop_indicies.append(index)
		transactions = transactions.drop(pop_indicies)
		return transactions


	'''After finding max minsup, we "threshold" the value such that
	we examine a larger subset of graphs. This helps us find relevant
	patterns that were not maximally frequent given minsup.'''
	def threshold_frequent_graphs(self, minsup):
		minsup = max(minsup - self.minsup_threshold, 0.01)
		return fpmax(self.transactions, minsup)

	'''Max minsup will likely generate too narrow of an explanation.
	We therefor threshold max minsup iteratively while delta F score
	is non-negative'''
	def create_weighted_explanation(self, max_minsup):
		#print ("        Starting WE")
		''' After F score diminishes, stop. Top F-score is presented as
		higher than its range so that it has work to do'''
		top_F_score = -1.0
		terminal = False
		graphs = []
		explanation = []
		transactions_cover = self.transactions.copy()
		prev = -1
		# Redunant list to ease computation
		explained = []
		# Step size: 1% or a single transaction (whichever greater)
		decrement = max(0.01, 1/len(self.transactions))
		minsup = max_minsup
		# Find explanations at various minsups
		while (minsup > 0 and not terminal):
			prediction = []
			#print ("        " + str(minsup))
			potential_graph_tups = fpmax(self.transactions, minsup)
			potential_graph_tups = potential_graph_tups.values.tolist()
			for tup in potential_graph_tups:
				graph = list(tup[1])
				if self.complete_graph(graph):
					prediction.append(graph)
			# To create F-score
			tp = 0
			fp = 0
			fn = 0
			tn = 0
			# Calculate F-score
			for entry in self.test_activations:
					for act in entry:
						if self.predicted(act, prediction):
							tp += 1
						else:
							fp += 1
					for p in prediction:
						for el in p:
							if el not in entry:
								fn += 1
						tn = len(self.test_activations) * self.threshold * \
						self.total_neurons - tp - fn - fp
			precision = tp / (tp + fp)
			recall = tp / (tp + fn)
			f = 2.0 / (1.0 / recall + 1.0 / precision)
			# Determine if terminal
			if f >= top_F_score:
				top_F_score = f
				graphs.append((prediction, minsup))
				minsup -= decrement
			else:
				terminal = True
		# Weight explanations
		k = True
		for ex in graphs:
			if k:
				k = False
			for graph in ex[0]:
				'''Ensures that only new parts of the graph are added
				to the explanation'''
				for el in graph:
					if el not in explained:
						explained.append(el)
						explanation.append((el, ex[1]/max_minsup))
		print ()
		return explanation

	'''Returns true is an activation is predicted as accurate.
	Abstracted into a method due to nested for-loops'''
	def predicted(self, activation, prediction):
		for p in prediction:
			for el in p:
				if activation == el:
					return True
		return False


	'''Find if a graph is a complete graph given the graph architecture'''
	def complete_graph(self, graph):
		# Check if we have a node from each layer
		layers = []
		for node in graph:
			layer = self.find_layer(node)
			if layer not in layers:
				layers.append(layer)
		return len(layers) == len(self.architecture)

	def run(self):
		minsup = self.find_max_minsup()
		self.explanation = self.create_weighted_explanation(minsup)
		return self.explanation

'''Explains a model by explaining each of its classes'''
class ModelExplanation():

	def __init__(self, architecture, threshold, weight_csv,
		valid_csv_file = None, invalid_csv_file = None, 
		confusion_csv_file = None, output_file = None, minsup_threshold = 0.0):
		'''Arguments:
		csv_file (string): filepath to a csv_file in the format:
		<<class label>>, <<neuron value 1>>, <<neuron value 2>>, ...
		(repeated for all instances)

		architecture (list of int): a list of ints where each int is
		the number of neurons in feed-forward, and the length of the
		list is the number of feed-forward layers.

		threshold (int): Threshold for activation. The threshold-
		highest neurons are considered activated, all other not.
		Range: [1 , number of neurons-1]

		minsuo_threshold (float): higher values mean that more nodes
		will be considered part of the explanation. Range: [0,1.0]'''

		if output_file != None and not os.path.isdir(os.path.dirname(output_file)):
			print ("ERROR: Output has no destination")

		self.architecture = architecture
		self.threshold = threshold
		self.minsup_threshold = minsup_threshold
		self.output_file = output_file
		self.class_explanations = {}
		self.invalid_class_explanations = {}
		self.confusion_class_explanations = {}
		self.class_neurons = {}
		self.test_neurons = {}
		self.invalid_class_neurons = {}
		self.invalid_test_neurons = {}
		self.confusion_class_neurons = {}
		self.confusion_test_neurons = {}

		with open(weight_csv, newline='') as block:
			weights = csv.reader(block, delimiter=',', quotechar='|')
			for row in weights:
				for i in range(len(row)):
					row[i] = float(row[i])
				self.weights = row
		# Get all neurons from input files
		if valid_csv_file != None:
			valid_neurons = self.read_neurons(valid_csv_file, delimiter=' ')
			for c in valid_neurons.keys():
				self.class_neurons[c] = valid_neurons[c]
				self.test_neurons[c] = valid_neurons[c]
		if invalid_csv_file != None:
			invalid_neurons = self.read_neurons(invalid_csv_file, delimiter=' ')
			for c in invalid_neurons.keys():
				self.invalid_class_neurons[c] = invalid_neurons[c]
				self.invalid_test_neurons[c] = invalid_neurons[c]
		if confusion_csv_file != None:
			confusion_neurons = self.read_neurons(confusion_csv_file, delimiter=',')
			for c in confusion_neurons.keys():
				self.confusion_class_neurons[c] = confusion_neurons[c]
				self.confusion_test_neurons[c] = confusion_neurons[c]


	def read_neurons(self, filepath, delimiter = ','):
		neurons = {}
		with open(filepath, newline='') as block:
			neuron_values = csv.reader(block, delimiter=delimiter,
				quotechar='|')
			for row in neuron_values:
				# Convert strings to floats
				for i in range(len(row) - 1):
					if row[i + 1] != '':
						row[i + 1] = float(row[i + 1])
				#row = [float(i) for i in row[0].split(',')]
				# Get all classes ready for evaluation
				if row[0] not in neurons.keys():
					neurons[row[0]] = []
				# Get each subset of neurons for each class
				neurons[row[0]].append(row[1:])
		return neurons

	'''The model confusion explanation is a 2D dictionary in the format:
	{true_name1: {incorrect_name1: [(neuron_x, weight1),
		(neuron_y, weight2), ...], incorrect_name2: [...], ...},
	true_name2: {...}}
	where each key is the correct name of a class, which has the
	corresponding value as a dictionary, in which the keys are the name
	of the class that was mispredicted, and the corresponding value is a
	frequent subgraph explanation'''
	def make_model_confusion_explanation(self):
		model_confusion_explanation = {}
		correct_names = []
		for c in self.confusion_class_explanations.keys():
			confusion = c.split(' ')
			correct_name = confusion[0]
			mispredicted_name = confusion[1]
			if mispredicted_name not in model_confusion_explanation.keys():
				model_confusion_explanation[mispredicted_name] = {}
			model_confusion_explanation[mispredicted_name][correct_name] = self.confusion_class_explanations[c]

		self.confusion_class_explanations = model_confusion_explanation

	# Remove overlap between valid and invalid explanations
	def make_invalid_explanation(self):
		for c in self.invalid_class_explanations.keys():
			valid_exp = self.class_explanations[c]
			invalid_exp = self.invalid_class_explanations[c]
			valid_neurons = []
			for el in valid_exp:
				valid_neurons.append(el[0])
			for el in invalid_exp:
				if el[0] in valid_neurons:
					invalid_exp.remove(el)


	# Explain each class, output subset of nodes
	def explain(self):
		print ("Starting valid explanation")
		for c in self.class_neurons.keys():
			#print ("    " + str(c))
			self.class_explanations[c] = None
			explanation = ClassExplanation(c, self.class_neurons[c],
				self.test_neurons[c], self.weights, self.architecture,
				self.threshold, minsup_threshold = self.minsup_threshold)
			self.class_explanations[c] = explanation.run()

		print ("Starting invalid explanation")
		for c in self.invalid_class_neurons.keys():
			print ("    " + str(c))
			self.invalid_class_explanations[c] = None
			explanation = ClassExplanation(c, self.invalid_class_neurons[c],
				self.invalid_test_neurons[c], self.weights, self.architecture,
				self.threshold, minsup_threshold = self.minsup_threshold)
			self.invalid_class_explanations[c] = explanation.run()
		self.make_invalid_explanation()

		print ("Starting confusion explanation")
		for c in self.confusion_class_neurons.keys():
			print ("    " + str(c))
			self.confusion_class_explanations[c] = None
			explanation = ClassExplanation(c, self.confusion_class_neurons[c],
				self.confusion_test_neurons[c], self.weights, self.architecture,
				self.threshold, minsup_threshold = self.minsup_threshold)
			self.confusion_class_explanations[c] = explanation.run()
		self.make_model_confusion_explanation()

	def save_output(self):
		# Convert explanations to strings
		class_explanation_str = str(self.class_explanations) + '\n'
		invalid_explanation_str = str(self.invalid_class_explanations) + '\n'
		confusion_explanation_str = str(self.confusion_class_explanations) + '\n'
		output_strs = [class_explanation_str, invalid_explanation_str, confusion_explanation_str]
		# Write to the specified output file
		f = open(self.output_file, 'w')
		f.writelines(output_strs)
		#Close file
		f.close()

	def clear_explanations(self):
		self.class_explanations = {}


#Driver code for the Labeled_Activation dataset

start_time = time.time()
for i_prime in range(4):
	i = i_prime + 2
	e = ModelExplanation([80, 60, 40, 30, 20], 50,
		os.path.join('..', 'Faces', 'weights', 'weights' + str(i) + '.csv'),
		valid_csv_file=os.path.join('..', 'Faces', 'activations', 'facial_training_true_' + str(i) + '.csv'),
		invalid_csv_file=os.path.join('..', 'Faces', 'activations', 'facial_training_false_' + str(i) + '.csv'),
		confusion_csv_file=os.path.join('..', 'Faces', 'activations', 'facial_confusion' + str(i) + '.csv'),
		output_file=os.path.join('..', 'Faces', 'explanations_cover', 'explanation' + str(i) + '.txt'))
	'''e = ModelExplanation([256, 32], 50,
		os.path.join('..', 'Bird', 'Weights', 'weights_' + str(i) + '.csv'),
		valid_csv_file=os.path.join('..', 'Bird', 'Activations', 'activation_' + str(i) + '_train_plus' + '.csv'),
		invalid_csv_file=os.path.join('..', 'Bird', 'Activations', 'activation_' + str(i) + '_train_minus' + '.csv'),
		output_file=os.path.join('..', 'Bird', 'Explanations', 'explanation' + str(i) + '.txt'))
	e = ModelExplanation([256, 32], 50,
		os.path.join('..', 'Bird', 'Weights', 'weights_' + str(i) + '.csv'),
		valid_csv_file=os.path.join('..', 'Bird', 'Activations', 'activation_' + str(i) + '_train_plus' + '.csv'),
		output_file=os.path.join('..', 'Bird', 'Explanations', 'explanation' + str(i) + '.txt'))'''
	e.explain()
	#e.save_output()
	end_time = time.time()

	print ("Computation time: " + str(end_time - start_time))












