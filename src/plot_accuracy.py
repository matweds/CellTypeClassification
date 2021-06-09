'''
File to plot different graphs and tables for the results
Authors: Mattias Wedin, Isak Bengtsson

Insperation
https://github.com/MarcusJoakimKex2018/AutomaticClassification
'''

from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
import json

from skimage.color.colorconv import rgb2gray, rgba2rgb

# Path to result unbalanced 
PATH_TO_RESULT_UB = '../data/result_after10_unbalanced.json'
PATH_TO_RESULT_ACC_UB = '../data/result_after10_accuracy_unbalanced'
PATH_TO_RESULT_IND_UB = '../data/result_after10_accuracy_individual_unbalanced.json'

# Path to result balanced
PATH_TO_RESULT_B = '../data/result_after10_balanced.json'
PATH_TO_RESULT_ACC_B = '../data/result_after10_accuracy_balanced'
PATH_TO_RESULT_IND_B = '../data/result_after10_accuracy_individual_unbalanced.json'

# TODO Fix command line argument for different plots 
# Change this to use different paths
data_path = Path(PATH_TO_RESULT_UB)
f = open(data_path)
data = json.load(f)
# Plot the accuracy for each model on each cell type
def plot_cell_acc():
    classifiers = [[] for i in range(3)]
    # path_ind = Path(PATH_TO_RESULT)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.boxplot(classifiers)

    ax.set_xticklabels(['SVM', 'Random Forest', 'CNN'], rotation=330)

    plt.grid()
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Model accuracy (%)")
    plt.show()
    
    # The cell type hit ratio diagrams
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout() 
    i = 1
    for model, cells in data.items():
        print(model)
        names = []
        count = 0
        values = []
        for k, v in sorted(cells.items()):
            names.append(count)
            count += 1
            if v[0] + v[1] == 0:
                v[1] = 1
            values.append((v[0]/(v[0]+v[1]))* 100)
        
        plt.xticks(rotation=0)
        plt.subplot(3, 1, i,)
        plt.ylim([0,100])
        plt.bar(names, values , width=0.5)
        plt.title(model)
        plt.ylabel('%')
        i+=1
    fig.set_size_inches(10,8)
    plt.savefig('../results/cell_type_acc_balanced', dpi=100)

def plot_cell_acc_box():
    cell_acc = {}
    classifiers = [[] for i in range(3)]
    # path_ind = Path(PATH_TO_RESULT_IND)
    path_ind = Path(PATH_TO_RESULT_IND_L)

    f = open(path_ind)
    data_ind = json.load(f)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.boxplot(classifiers)

    ax.set_xticklabels(['SVM', 'Random Forest', 'CNN'], rotation=330)

    plt.grid()
    plt.title("Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Model accuracy (%)")
    plt.show()
    
    # The cell type hit ratio diagrams
    fig, axes = plt.subplots(nrows=3, ncols=1)
    fig.tight_layout() 
    i = 1
    values = {}
    for model, run in data_ind.items():
        print(model)
        names = []
        values[model] = {}
        count = 0
        for r , cells in run.items():
            for cell,_ in cells.items():
                values[model][cell] = []
        for cell, _ in sorted(run['0'].items()):
            print(cell)
            names.append(cell)
        for k, v in sorted(run.items()):
            for c, val in v.items():
                count += 1
                if v[c][0] + v[c][1] == 0:
                    v[c][1] = 1
                values[model][c].append((v[c][0]/(v[c][0]+v[c][1]))* 100)

        all_values = []
        
        for cell, val in sorted(values[model].items()):
            all_values.append(val)
        print(all_values)
        # print(values)
        plt.xticks(rotation=0)
        plt.subplot(3, 1, i)
        plt.ylim([0,100])
        plt.boxplot(all_values, labels=names)
        
        # plt.bar(names, values, width=0.5)
        plt.title(model)
        plt.ylabel('%')
        i+=1
    fig.set_size_inches(10,8)
    plt.savefig('../results/cell_type_acc_boxplot_balanced', dpi=100)

def plot_acc_boxplot():
    mean = []

    # Use boxplot to plot standard deviation
    data_array = []
    for model, acc in data.items():
        print(acc)
        data_array.append(acc)

    # ax = fig.add_axes([0, 0, 1, 1])
    fig = plt.figure(1, figsize=(9, 6))

    ax = fig.add_subplot(111)
    bp = ax.boxplot(data_array)

    ax.set_xticklabels(['SVM', 'Random Forest', 'CNN'])
    ax.set_ylim([0,1])
    plt.grid()
    plt.legend()
    plt.title("Model Comparison")
    plt.ylabel("Model accuracy (%)")
    plt.savefig('../results/mean_acc_boxplot_balanced.png')

def plot_acc_table():
    mean = []

    # Plot overall accuracy for each model as a table.

    for model, acc in data.items():
        print(acc)
        mean.append([model,round(np.mean(acc)*100, 3)])
        print(np.mean(acc))

    print(mean)
    fig, ax =plt.subplots()
    # hide axes
    fig.patch.set_visible(False)

    ax.axis('off')
    ax.axis('tight')
   
    

    collabel = ('Classifier', 'Mean Accuracy')
    ax.table(cellText=mean,colLabels=collabel,loc='center')

    plt.savefig('../results/mean_acc_balanced_table.png')

def plot_images():
    img_path = Path('../data/images/mouse_new_filtered/')
    save_path = Path('../data/images/samples/')
    dimension = (300,300)

    images = {}
    # Get images
    mediumS_img = imread(img_path.joinpath('medium spiny/2C5_1.png'))
    purkinje_img = imread(img_path.joinpath('Purkinje/10_f1_d1-1_1b.png'))
    pyramidal_img = imread(img_path.joinpath('pyramidal/202522.png'))

    mediumS_img = resize(mediumS_img, dimension, anti_aliasing=True, mode='reflect')
    purkinje_img = resize(purkinje_img, dimension, anti_aliasing=True, mode='reflect')
    pyramidal_img = resize(pyramidal_img, dimension, anti_aliasing=True, mode='reflect')

    # Make grayscaled
    gray_mediumS_img = rgb2gray(rgba2rgb(mediumS_img))
    gray_pyramidal_img  = rgb2gray(rgba2rgb(pyramidal_img))
    gray_purkinje_img = rgb2gray(rgba2rgb(purkinje_img))

    images['Medium spiny'] = [mediumS_img, gray_mediumS_img]
    images['Purkinje'] = [purkinje_img, gray_purkinje_img]
    images['Pyramidal'] = [pyramidal_img, gray_pyramidal_img]

    # For debugging
    print(images)
    # Save images to file
    # imsave(save_path.joinpath('Medium_Spiny.png'), mediumS_img)
    # imsave(save_path.joinpath('Purkinje.png'), purkinje_img)
    # imsave(save_path.joinpath('Pyramidal.png'), pyramidal_img) 
    # imsave(save_path.joinpath('Medium_Spiny_gray.png'), gray_mediumS_img)
    # imsave(save_path.joinpath('Purkinje_gray.png'), gray_purkinje_img)
    # imsave(save_path.joinpath('Pyramidal_gray.png'), gray_pyramidal_img)


    plt.figure(figsize=(10, 10))
    i = 1
    for key in images.keys():
        for image in images[key]:
            grayscaled = ''
            ax = plt.subplot(3,2, i)
            if i % 2 == 0:
                plt.imshow(image, cmap='gray')
                grayscaled = ' grayscaled'
            else:
                plt.imshow(image)
            plt.title(key + grayscaled)
            plt.axis('off')
            i += 1
    plt.savefig("../data/images/samples/allSamples.png")


def plot_celltypes():
    nr_of_each =[]
    # Plot each cell, id and how many evaluated of each cell
    id = 0
    a = 0
    for model, cells in data.items():
        a+=1
        if a == 3:
            for k, v in sorted(cells.items()):
                total = v[0] + v[1]
                nr_of_each.append([k,total])
                id += 1
            break

    print(nr_of_each)
    nr_of_each.append([0,0])
    plt.figure()

    fig, ax =plt.subplots(1, 2, figsize=(15, 15))
    
    # hide axes
    fig.patch.set_visible(False)
    ax[0].axis('off')
    ax[1].axis('off')

    fig.tight_layout(pad=2)
    plt.rc('font', size=(15))
    collabel = ('Cell Type', 'Nr of validated cells')
    i = 0
    

    ax[0].table(rowLabels=range(25), cellText=nr_of_each[:25],colLabels=collabel, loc='center')
    ax[1].table(rowLabels=range(25,50),cellText=nr_of_each[25:],colLabels=collabel, loc='center')
    plt.savefig('../results/celltypes_new_cnn.png')

if __name__ == '__main__':
    # Change this to plot different plots

    # plot_images()
    # plot_acc_boxplot()
    # plot_cell_acc_box()  
    # plot_acc_table()  
    # plot_celltypes()
    plot_cell_acc()
