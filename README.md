# Identification and Uses of Deep Learning Backbones via Pattern Mining
## Michael Livanos & Ian Davidson
## University of California, Davis
## Published at SIAM SDM 2024

This repository contains the following files, with bolded entries being of particular importance, and the rest existing primarily for reproducibility purposes:

BirdExplanations.zip - Contains the prebuild explanations for the bird audio detection challenge networks. 

BirdNetwork.zip - Contains the networks (as .h5 files) for the bird audio detection challenge. One file for each of the 10 folds

BirdWeights.zip - Contains the weights as csv's, needed to create explanations. One file for each of the 10 folds

BirdActivations - Contains activation vectors (as csv's), one for correctly predicted instances, and another for incorrectly predicted instances, for each of the ten folds (20 files in total) NOTE: This file was too big for GitHub (~412MB) and could not be uploaded. It can be recreated using the getActivations script.

### **modelExplanation.py - Python3 file for generating backbones using our heuristic-based method, as described in Section 4 of the paper.**

ILP.py - Python3 file for running the relaxed ILP to generate backbones as described in Section 3 of the paper.

### **ExplanationAugmentedPredictor.py - Python3 file for augmenting predictions using the explanation**

### **getActivations.py - Python3 file to get the activation vectors using a network stored as a .h5 file and the training instances. Please note that the activations are already provided, but if you would like to generate them youself, you are free to do so.**

LFWActivation.zip - Contains the activation vectors (as csv's), one for correctly predicted instances, another for incorrectly predicted instances, and another for confusion, for each of the five folds (15 files in total)

LFWExplanations.zip - Contains all prebuilt explanations for the LFW networks

LFWNetworks.zip - Contains all of the networks dscribed in the paper, one network for each of the five folds

LFWWeights.zip - Contains all of the weights for each of the five folds as a cvs (needed to create the explanations)
