import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils.multiclass import unique_labels



def plot_confusion_matrix(results, class_names, label_id, label_name):
	PATH_OUTPUT = '../plots/'
	image_path = os.path.join(PATH_OUTPUT, 'Confusion_Matrix_'+label_name+'.png')

	y_true, y_pred = zip(*results)
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	y_true = y_true[:,label_id]
	y_pred = y_pred[:,label_id]
	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	# Only use the labels that appear in the data
	#class_names = class_names[unique_labels(y_true, y_pred)]
	
	np.set_printoptions(precision=2)
	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]), 
		yticks=np.arange(cm.shape[0]),
		# ... and label them with the respective list entries
		xticklabels=class_names, yticklabels=class_names,
		title='Normalized Confusion Matrix\n' + label_name,
		ylabel='True',
		xlabel='Predicted')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
		rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt), 
				ha="center", va="center", 
				color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()

	plt.savefig(image_path)


def plot_roc(targets, probs, label_names):
	PATH_OUTPUT = '../plots/'
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	font = {'size' : 15}
	plt.rc('font', **font)
	fig = plt.figure(figsize=(6,6))
	image_path = os.path.join(PATH_OUTPUT, 'ROC.png')
	for i, label_name in enumerate(label_names): # i th observation
		y_true = targets[:,i]
		y_score = probs[:,i]

		# drop uncertain
		iwant = y_true < 2
		y_true = y_true[iwant]
		y_score = y_score[iwant]	
		
		fpr[i], tpr[i], _ = roc_curve(y_true, y_score)
		roc_auc[i] = auc(fpr[i], tpr[i])
		
		plt.subplot(1, 1, i+1)
		plt.plot(fpr[i], tpr[i], color='b', lw=2, label='ROC (AUC = %0.2f)' % roc_auc[i])
		plt.plot([0, 1], [0, 1], 'r--')
		plt.xlim([0, 1])
		plt.ylim([0, 1.0])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title(label_name)
		plt.legend(loc="lower right")

	plt.tight_layout()
	fig_size = plt.rcParams["figure.figsize"]
	fig_size[0] = 30
	fig_size[1] = 10
	plt.rcParams["figure.figsize"] = fig_size
	#plt.savefig(image_path)

