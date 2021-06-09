# KEX VT21 
Automatic classification of mouse cell types from digitally reconstructed images
### Code
#### data_collection.py
Collects the data from neuromorpho.org's api

* Images of mouse neurons [NeuroMorpho](http://neuromorpho.org/)


#### image_refactor.py
Deletes broken images from the collected data

#### classification.py
Creates the CNN and runs the test using the CNN

#### svm_rf_classification.py
Creates the SVC and RF and runs the test

#### plot_accuracy.py
Plots different accuracy tables and diagrams

### Data
The result from the tests in form of json

### results
The plots from the final result used in the report
