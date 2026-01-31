# Plant Disease Image Classifier training & Model compression with quantization and pruning (in PyTorch) 
A PyTorch project for plant disease classification with images, along with model compression through quantization and pruning.

A demo is also available [here](https://huggingface.co/spaces/edmos7/plant-leaf-classifier), where you can upload an image and get predictions from the quantized model.

The project contains different files, explained below (the files are also ordered as intended for project inspection)

- exploration.ipynb : The dataset for plant leaves images is first loaded and inspected to get acquainted with its properties
- intensity.ipynb : An analysis of image intensity across the dataset 
- model.py : A .py file containing model classes used for in the subsequent files for fitting
- training.ipynb : training and testing of a convolutional model for plant leaf classification
- compression.ipynb : the trained model is compressed via pruning and quantization, and the results evaluated
---> models folder : contains model checkpoints corresponding to:
	- best_model.pth : the image classfier checkpoint for the best (lowest val_loss) epoch in training (from training.ipynb)
	- pruned_model.pth : the globally pruned (50%) model obtained from best_model (from compression.ipynb)
	- pruned_finetuned_model.pth : new version of pruned_model finetuned on the new architecture
	- _fbgemm_quantized_model_dict.pth : the model dictionary for the quantized version of best_model, using fbgemm as backend engine
	- _fbgemm_pruned_finetuned_model_dict.pth : the quantized version of pruned_finetuned_model.

The notebooks can run in colab, to do so you should add a this project to your drive along with the dataset(you will have to modify a couple of path string variables), further details are contained in the comments of the notebooks. Alternatively, in order to run everything locally, clone the repo and download the dataset (you will also have to modify a couple of path string variables).

Dataset available here: https://data.mendeley.com/datasets/tywbtsjrjv/1

Dataset paper available here: https://arxiv.org/pdf/1511.08060

@article{Mohanty_Hughes_Salathé_2016,
title={Using deep learning for image-based plant disease detection},
volume={7},
DOI={10.3389/fpls.2016.01419},
journal={Frontiers in Plant Science},
author={Mohanty, Sharada P. and Hughes, David P. and Salathé, Marcel},
year={2016},
month={Sep}} 
