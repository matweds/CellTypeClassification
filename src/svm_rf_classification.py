'''
Image classification using SVM and random forest

Authors: Isak Bengtsson, Mattias Wedin

Sources:
https://github.com/whimian/SVM-Image-Classification/blob/master/Image%20Classification%20using%20scikit-learn.ipynb
https://github.com/PraveenDubba/Image-Classification-using-Random-Forest/blob/master/Random_Forest_latest.py
'''


from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, metrics
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


import skimage.io
from skimage.color import rgb2gray
from skimage.transform import resize

import json

# Set memory limit
# def limit_memory(maxsize):
#     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#     resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))

# mem = 14000000 * 1024
# limit_memory(mem)

def load_img_files(container_path, dimension=(300, 300)):
    '''
    Load and prepares images for SVM and RF
    '''
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    descr = "A image classification dataset"
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        counter = 0
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            img_resized = rgb2gray(img_resized)
            flat_data.append(img_resized.flatten()) 
            target.append(i)
            # if(counter%500 == 0):
            #     print(counter)
            # if(counter == 10):
            #     break
            counter+=1
    flat_data = np.array(flat_data, dtype=object)
    target = np.array(target)

    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 DESCR=descr)


## Get the dataset
# Dataset for balanced data
image_dataset = load_img_files("../data/images/mouse_new_filtered/")

# Dataset for unbalanced data
# image_dataset = load_img_files("../data/images/mouse/")


# Split the data into train and validation
# X_train, X_test, y_train, y_test = train_test_split(
#     image_dataset.data, image_dataset.target, test_size=0.3)


# Train the data with parameter optimizations svm 
def svm_classifier():
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'degree':[2,3], 'kernel': ['poly'], 'tol':[1e-3, 1e-4], 'coef0':[100, 10]}
    ]

    from sklearn.metrics import f1_score
    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()
        clf = GridSearchCV(
            SVC(), param_grid,cv=3 , scoring='%s_macro' % score
        )
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        metrics.f1_score(y_test, y_pred, labels=np.unique(y_pred), average=None)
        print(metrics.classification_report(y_true, y_pred))
        print()

# Run svm classifier
#svm_classifier()


# Random forest classifier, this is used to find an tuned model.
def rf_classifier():
    forest_clf = RandomForestClassifier(random_state=42, verbose=True)

    # Not tuned in any way
    forest_clf.fit(X_train, y_train)
    y_pred_train = forest_clf.predict(X_train)
    y_pred_test =  forest_clf.predict(X_test)
    print("Training metrics:")
    print(metrics.classification_report(y_true= y_train, y_pred= y_pred_train))

    print("Test data metrics:")
    print(metrics.classification_report(y_true= y_test, y_pred= y_pred_test))
    # 0.67% accuracy

    # Using Grid search for hyper parameter tuning
    print("\n ======= White gridseach =======")
    clf = GridSearchCV(forest_clf, param_grid={'n_estimators':[100,200],'min_samples_leaf':[2,3]})
    model = clf.fit(X_train,y_train)
    
    model.fit(X_train, y_train)
    print(clf.best_params_)
    y_pred_train = model.predict(X_train)
    y_pred_test =  model.predict(X_test)
    print("Training metrics:")
    print(metrics.classification_report(y_true= y_train, y_pred= y_pred_train))

    print("Test data metrics:")
    print(metrics.classification_report(y_true= y_test, y_pred= y_pred_test))

# rf_classifier()

# The full test

RUNS = 10

# The classifiers used for final test
classifiers_name = ['SVC', 'Random Forest', 'CNN']
classifiers = [svm.SVC(C=100, gamma=0.001, kernel='rbf'),
RandomForestClassifier(min_samples_leaf=3, n_estimators=200)]

print(image_dataset.target_names)

# Scores
from sklearn.metrics import accuracy_score
scores = {}
for clf in classifiers_name:
    scores[clf] = []
scoreRatios = dict()
individualScoreRatios = dict()

for i in range(len(classifiers_name)):
    scoreRatios[classifiers_name[i]] = dict()
    individualScoreRatios[classifiers_name[i]] = dict()
    for k in range(RUNS):
        individualScoreRatios[classifiers_name[i]][k] = dict()
        for j in range(len(image_dataset.target_names)):
            individualScoreRatios[classifiers_name[i]][k][image_dataset.target_names[j]] = [0 , 0]
    for j in range(len(image_dataset.target_names)):
        scoreRatios[classifiers_name[i]][image_dataset.target_names[j]] = [0 , 0]
        


for it in range(RUNS): # amount of runs for each classifiers.
    # Splitting the dataset into the Training set and Test set
    # Split the data into train and validation
    print(it)
    X_train, X_test, y_train, y_test = train_test_split(
        image_dataset.data, image_dataset.target, test_size=0.3)

    for i in range(len(classifiers)):
        classifiers[i].fit(X_train, y_train)
        y_predictions = classifiers[i].predict(X_test)
        scores[classifiers_name[i]].append(accuracy_score(y_test, y_predictions))
        for j in range(len(y_test)):
            if y_test[j] == y_predictions[j]:
                scoreRatios[classifiers_name[i]][image_dataset.target_names[y_test[j]]][0] += 1
                individualScoreRatios[classifiers_name[i]][it][image_dataset.target_names[y_test[j]]][0] += 1
            else:
                scoreRatios[classifiers_name[i]][image_dataset.target_names[y_test[j]]][1] += 1
                individualScoreRatios[classifiers_name[i]][it][image_dataset.target_names[y_test[j]]][1] += 1

        
def convert(o):
    if isinstance(o, np.int64): return int(o)  
    raise TypeError


with open('../data/result_after10_new_test_balanced.json', 'w', encoding='utf-8') as outfile:
    json.dump(scoreRatios, outfile, ensure_ascii=False, indent=4 ,default=convert)

with open('../data/result_after10_accuracy_test_balanced.json', 'w', encoding='utf-8') as outfile:
    json.dump(scores, outfile, ensure_ascii=False, indent=4 ,default=convert)

with open('../data/result_after10_accuracy_individual_balanced.json', 'w', encoding='utf-8') as outfile:
    json.dump(individualScoreRatios, outfile, ensure_ascii=False, indent=4 ,default=convert)

