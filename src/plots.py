import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# TODO: You can use other packages if you want, e.g., Numpy, Scikit-learn, etc.
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.utils.multiclass import unique_labels

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

