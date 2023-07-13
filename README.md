# A-Sheaf-Theoretic-Perspective-for-Robust-Segmentation

This is a repo for our work on A-Sheaf-Theoretic-Perspective-for-Robust-Segmentation

## Description

This repo contains the code for training robust segmentation models by enforcing shape equivariance in a discrete latent space and using cellular sheaf theory to model compositionality of the topology of the output segmentation map and enforce a compositional based loss (see paper for more details).
This codebase contains training and model code for our models. We have different types of models. We have models which enforce equivariance using a contrastive based loss as described in our paper. We enforce equivariance to the dihedral group (D4) using our contrastive base loss. We also enforce equivariance by constraining the convolutional kernels in our model to either regular or irreducible group representation.

## Getting Started

### Dependencies

* Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### Datasets

* You will need to create 3 csv files (train.csv, validation.csv, test.csv). The train.csv should have three colums ('t2image','adcimage','t2label') containing the paths to the images and 
corresponding segmentations. The validation.csv and test.csv should have two colums ('t2image','t2label') containing the paths to the images and 
corresponding segmentations. We support nifti format. We provide an example for Prostate data in data/Prostate.
* You are free to choose to train on the dataset of your choice, pre-processed as you wish. We have provide dataloaders for the prostate datasets.
  * * Prostate: The prostate dataset is acquired from the [NCI-ISBI13](https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures) Challenge and [decathalon dataset](http://medicaldecathlon.com/).

### Training/Testing.
* You can run the training/testing script together with main.py. You must enter the paths to the train, validation and test csv files and the output 
directory to save results and images. You will need to adjust other hyper-parameters according to your dataset which can be seen in main.py. We have 4 models:'ShapeVQUnet', 'HybridShapeVQUnet', 'HybridSE3VQUnet', '3DSE3VQUnet'. The 'ShapeVQUnet' and 'HybridShapeVQUnet' model constrains the latent space to a equivariant shape space to the D4 group using a contrastive based loss. You should choose the arguement --contrastive True if you choose the 'ShapeVQUnet' or 'HybridShapeVQUnet' model and choose --contrastive False otherwise. The 'ShapeVQUnet' is a 3D model while the 'HybridShapeVQUnet' is a 2D/3D model. The 'HybridSE3VQUnet' and '3DSE3VQUnet' model constrain the convolutionals kernels to the SE3 group. If you choose one the 'HybridSE3VQUnet' and '3DSE3VQUnet', you will have to choose whether you want a regular ('Regular') or irreducible ('Irreducible') group representation (--repr) . If you choose a regular ('Regular') group representation, then you will have to choose the group (--group) e.g. --group 4 is equivariance to the D4 group. You must also choose the multiplicity (--multiplicity) of each element in the group if one chooses the 'HybridSE3VQUnet' and '3DSE3VQUnet' model. For all models, you must also choose after how many epochs you want to include the cellular sheaf based loss (--topo_epoch) 
Below is an example for Prostate data
```
python main.py --modeltype 'HybridShapeVQUnet' --contrastive True --topo_epoch 25 --training_data '.../Sheaves_for_Segmentation/data/Prostate/train.csv' --validation_data '.../Sheaves_for_Segmentation/data/Prostate/validation.csv' --test_data '.../Sheaves_for_Segmentation/data/Prostate/test.csv', --output_directory '.../Sheaves_for_Segmentation/data/Prostate/output/'
```

## Authors

Contributors names and contact info

Ainkaran Santhirasekaram (a.santhirasekaram19@imperial.ac.uk)
![image](https://github.com/AinkaranSanthi/A-Sheaf-Theoretic-Perspective-for-Robust-Segmentation/assets/93624569/d645e3dd-b65c-4b27-b508-b1a84a5e294c)
