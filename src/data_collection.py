'''
Program to collect morphology metadata and images from neuromorpho.org

Authors: Isak Bengtson, Mattias Wedin
'''
import urllib.request, json
from PIL import Image
import requests
from io import BytesIO
import os
import matplotlib.pyplot as plt
import re

# URL to the neuromorpho api
MORPHO_URL = "http://neuromorpho.org/api/neuron/select?" 

#Morphologie json file
MORPHOLOGIE_DATA_FILE_MOUSE = "../data/mouse.json"
MOUSE_IMAGES = "../data/images/mouse_new/"


"""
Get the morphologie metadata from neuromorpho.org:s api and save to file
"""
def get_morhpology(species, brain_region, path):

    # Query params. Not fully decided yet 
    #species = "mouse"
    #brain_region = "neocortex"

    pageNumber = 0
    # This only fetches the first page, i.e. the first 50 data
    query = "http://neuromorpho.org/api/neuron/select?page="+str(pageNumber)+"&q=species:"+species#+"&fq=brain_region:"+brain_region
    print(query)
    with urllib.request.urlopen(query) as url:
        data = json.loads(url.read().decode())
        # Get number of pages, i.e. all the data
        number_of_pages = data["page"]["totalPages"]
        print(number_of_pages)
        
        # get all the pages
        for i in range(number_of_pages-1):
            print(str(i))
            query = "http://neuromorpho.org/api/neuron/select?page="+str(i+1)+"&q=species:"+species#+"&fq=brain_region:"+brain_region
            print(query)
            with urllib.request.urlopen(query) as url:
                new_data = json.loads(url.read().decode())
                data["_embedded"]["neuronResources"].extend(new_data["_embedded"]["neuronResources"])
        
        # dump to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

'''
Get image for each morphologie and save to file as .png
Name structure: {species}_{neuron_name}.png
'''
def get_image(data_path, image_path):

    # All cell types, used when 
    """ cell_types = ['pyramidal', 'granule', 'medium spiny', 'sensory receptor',
                    'bipolar',
                    'ganglion', 'amacrine','dopaminergic',
                    'Motoneuron', 'Somatostatin (SOM)-positive',
                    'Parvalbumin (PV)-positive OR basket OR Fast-spiking',
                    'Neuropeptide Y (NPY)-positive', 'Martinotti','Purkinje','thalamocortical',
                    'stellate', 'prion protein (PrP) promoter-positive',
                    'Vasoactive Intestinal Peptide (VIP)-positive',
                    'Steroid Hormone Receptor Ad4BP-positive',
                    'callosal-projecting','periglomerular','mitral',
                    'Cajal-Retzius', 'Serotonin receptor type 3A(5HT3)-positive',
                    'tufted', 'neurogliaform', 'Plasma Retinol-Binding Protein-Positive',
                    'bitufted', 'tyrosine-hydroxylase-positive', 'Golgi' ,'translocating',
                    'low-threshold calcium spiking', 'Chandelier', 'Thick-tufted',
                    'shrub cell', 'corticothalamic', 'horizontally elongated',
                    'Neuronal Nitric Oxide Synthase (nNOS)-positive',
                    'deep projecting cell', 'single bouquet', 'Midbrain-projecting'] 
    """
    

    # Domain no axon, only dendrites
    NO_AXON = 'No Axon'

    # Specific cell types for test 2
    # Cell types and their location
    cell_types = {'pyramidal': ['neocortex'], 'medium spiny': ['striatum'], 'Purkinje': ['cerebellum']}

    # open morphology file
    f = open(data_path)
    print(data_path)
    data = json.load(f)
    # species = ""
    png_url = ""
    # Iterate trough the data and get name,cell type and png url
    # Currently range(N) and not all, just for testing purposes
    print(len(data["_embedded"]["neuronResources"]))
    for i in range(len(data["_embedded"]["neuronResources"])):
        # if(i%1000 == 0 and i != 0):
        #     print("#"+ str(i), end="", flush=True)
        pyramidal = False
        found_type = False
        cell_type = ''

        neuron_name = data["_embedded"]["neuronResources"][i]["neuron_name"]
        brain_region = data["_embedded"]["neuronResources"][i]["brain_region"]
        domain = data["_embedded"]["neuronResources"][i]["domain"]
        # There may be multiple celltype
        types = data["_embedded"]["neuronResources"][i]["cell_type"]


        # TODO Fix command line argument to choose if download specific or all

        # Collection all images and put in directories
        """ for t in range(len(types)):
            if(types[t] == 'pyramidal'):
                pyramidal = True
            elif types[t] in cell_types:
                cell_type = types[t]
                found_type = True
                break

            if(pyramidal and not found_type):
                cell_type = 'pyramidal_'+brain_region[0]
            elif not found_type:
                cell_type = 'Others'

            png_url = data["_embedded"]["neuronResources"][i]["png_url"]
            img_path = os.path.join(image_path, cell_type)
            # Get image from url and save to file
            with requests.get(png_url) as response:
                # Check if dir exists
                if(not os.path.exists(img_path)):
                    os.mkdir(img_path) 
                file = open(img_path + "/" + neuron_name + ".png", "wb")
                file.write(response.content)
                file.close
        """
        #  Collection balanced dataset and put in directories
        # If not No Axon, do not use cell
        if(re.search(NO_AXON, domain) != None):
            if(types != None):
                for t in range(len(types)):
                    if types[t] in cell_types:
                        for region in brain_region:
                            if region in cell_types[types[t]]:
                                cell_type = types[t]

                            if cell_type != '':
                                png_url = data["_embedded"]["neuronResources"][i]["png_url"]
                                #print("Getting image", cell_type)
                                img_path = os.path.join(image_path, cell_type)
                                # Get image from url and save to file
                                with requests.get(png_url) as response:
                                    # Check if dir exists
                                    if(not os.path.exists(img_path)):
                                        os.mkdir(img_path) 
                                    file = open(img_path + "/" + neuron_name + ".png", "wb")
                                    file.write(response.content)
                                    file.close
                                break


import pathlib
# Probablby need to refactor images to png
# def refactor_images():
#     data_dir = pathlib.Path(PATH_TO_DATA)

def count_data(path, output_path):
    # TODO add command line argument
    """ 
        Count the number of images in each subdirectory in a directory. 
    """
    bad_images=[]
    number_of_images = []
    types = []
    # TODO change this to use pathdir
    s_dir = pathlib.Path(path)
    # s_dir = r'/home/mattias/Skola/KEX/KEX_VT_21/data/images/mouse_new_filtered'
    s_dir = r'/home/mattias/Skola/KEX/KEX_VT_21/data/images/mouse'
    bad_ext=[]
    total = 0
    s_list= os.listdir(s_dir)
    for klass in sorted(s_list):
        klass_path=os.path.join (s_dir, klass)
        # print ('processing class directory ', klass)
        if os.path.isdir(klass_path):
            file_list=os.listdir(klass_path)
            if(len(file_list) > 5):
                number_of_images.append(len(file_list))
                types.append(klass)
                total += len(file_list)
            else:
                print(klass)

    print(len(types))
    print(number_of_images)
    # print(number_of_images)
    # print(types)
    # Plot distribution
    plt.rc('font', size=6.5)
    plt.xticks(rotation=-90)
    # plt.gcf().set_size_inches(6, 3, forward=True)
    plt.gcf().subplots_adjust(bottom=0.50)
    plt.bar(types, number_of_images, align='center')#, width=0.2)
    plt.title("Cell type distribution")
    plt.ylabel('Number of cells')
    plt.savefig(output_path, dpi=200)

if __name__ == "__main__":
    count_data( '../data/data_distribution_unbalanced.png')
    count_data('../data/data_distribution_unbalanced.png')
    # get_morhpology("mouse", "neocortex", MORPHOLOGIE_DATA_FILE_MOUSE)

    # Get images, now only 10 each
    # get_image(MORPHOLOGIE_DATA_FILE_MOUSE, MOUSE_IMAGES)
    #get_image(MORPHOLOGIE_DATA_FILE_RAT, RAT_IMAGES)
    #get_image(MORPHOLOGIE_DATA_FILE_MOUSE, MOUSE_IMAGES)
    # refactor_images()


