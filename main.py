import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import os
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models import *
from dataloader import ProstateMRIDataModule
from traincontrastiveD4 import TrainD4
from traincontrastiveSE3 import TrainSE3
from utils import setup_distributed
from torch.nn.parallel.distributed import DistributedDataParallel
def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def none_or_int(value):
    if value == 'None':
        return None
    return value

def get_args_parser():
    parser = argparse.ArgumentParser('Vector Quantisation  training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--topo_epoch', type=int,  default=10,
                        help='Epoch from which to include the topological loss')

    # Model parameters
    parser.add_argument('--modeltype', default='HybridShapeVQUnet', type=str, choices=['ShapeVQUnet', 'HybridShapeVQUnet', 'HybridSE3VQUnet', '3DSE3VQUnet'],
                        help='Name of model to train')
    parser.add_argument('--contrastive', type=str_to_bool, default=True,
                        help='whether to apply contrastive based equivariance or constrain the convolutional kernels equivariant linear function')
    parser.add_argument('--image_size',  nargs="+",type=int, default=[256, 256, 24],
                        help='Size of input into the model')
    parser.add_argument('--patch_size',  nargs="+",type=int, default=[128, 128, 24],
                        help='Patch size to divide image into in order to build the cellular sheaf')
    parser.add_argument('--dim', default='3D', type=str,
                        help='Dimension of image input')
    parser.add_argument('--ch_3D', type=int,  default=1,
                        help='If you are using a hybrid model, then how many levels in the encoder/decoder do you want to consist of 3D convolutions')
    parser.add_argument('--dropout', type=float, default=0.0, 
                        help='Dropout rate for convolutional blocks (default: 0.)')
    parser.add_argument('--groups', type=int,  default=4, 
                        help='The number of groups for grouped normalisation. If None then batch normalisation')
    parser.add_argument('--in_ch', type=int,  default=1,
                        help='The number of input channels')
    parser.add_argument('--out_ch', type=int,  default=3,
                        help='The number of output channels')
    parser.add_argument('--channels', type=int,  default=12,
                        help='The number of channels from first level of encoder')
    parser.add_argument('--enc_blocks' , nargs="+",type=int, default=[1,1,1, 1, 1],
                        help='Number of ResBlocks per level of the encoder')
    parser.add_argument('--dec_blocks', nargs="+",type=int, default=[1, 1,1, 1],
            help='Number of ResBlocks per level of the decoder')
    parser.add_argument('--act', default='nn.LeakyReLU(0.2)', type=str,
                        help='Activation function to use. Enter "swish" if swish activation or enter function in torch function format if required different activation i.e. "nn.ReLU()"')
    parser.add_argument('--with_conv', type=str_to_bool, default=False,
                        help='Applying upsampling with convolution')
    parser.add_argument('--VQ', type=str_to_bool, default=True,
                        help='Apply vector quantisation in the bottleneck of the architecture. If False, then turns into original architecture')
    parser.add_argument('--quantise', default='spatial', type=str, choices=['spatial', 'channel'],
                        help='Quantise either spatially or channel wise, enter either "spatial" or "channel"')
    parser.add_argument('--n_e', default=128, type=int,
                        help='The number of codebook vectors')
    parser.add_argument('--embed_dim', default=192, type=int,
                        help='The number of channels before quantisation. If quantising spatially, this will be equivelent to the codebook dimension')
    parser.add_argument('--compute_ph', default='manual', type=str, choices=['manual', 'ripser'],
                        help='Whether compute the betti numbers of your label manually ripser. If your label is just one connected component with no holes or voids, then it is better to compute the betti numbers manually')
    parser.add_argument('--topo_fg_only', type=str_to_bool, default=True,
                        help='Whether to apply the topological loss to only the foreground (True) or the seperate foreground classes(False)')
    parser.add_argument('--repr', default='Regular', type=str, choices=['Regular', 'Irreducible'],
                        help='Whether to use Regular or Irreducible group representation if choose equivariant constrained cnn model')
    parser.add_argument('--group', type=int,  default=4, 
                        help='If choose to Regular group representation, then which group do you what do you want equivariance to. For example the dihedral group (D4), choose integer 4')
    parser.add_argument('--multiplicity' , nargs="+",type=int, default=[2,3,6,12],
                        help='If using equivariant constrained cnn model, then what multiplicity of each group representation at each level of the encoder/decoder do you choose?')
    parser.add_argument('--multiplicity3D' , nargs="+",type=int, default=[12],
                        help='If using Hybrid equivariant constrained cnn model, then what multiplicity of the encoder/decoder3D levels containing SE3 group representations  do you choose?')
    parser.add_argument('--num_features' , nargs="+",type=int, default=[2,3,3,3],
                        help='If using se3 equivariant constrained cnn model with irreducible group representation, then how what depth of group representations do you choose?')
    parser.add_argument('--se3channel' , nargs="+",type=int, default=[200, 480, 480, 960],
                        help='If using se3 equivariant constrained cnn model with irreducible group representation, then how many channels at each level of the encoder/decoder do you choose?')
    parser.add_argument('--init', default='he', type=str, choices=['he', 'delta', 'rand'],
                        help='How do you want to initialise the weights in your models')

    # Dataset parameters
    parser.add_argument('--dataset', default='prostate', type=str, choices=['prostate', 'abdomen', 'chest'],
                        help='Pick dataset to use, this can be extended if the user wishes to create their own dataloaders.py file')
    parser.add_argument('--training_data', default='.../Sheaves_for_Segmentation/data/Prostate/train.csv', type=str, required = False,  
                        help='training data csv file')
    parser.add_argument('--validation_data', default='.../Sheaves_for_Segmentation/data/Prostate/validation.csv', type=str, required = False,
                        help='validation data csv file')
    parser.add_argument('--test_data', default='.../Sheaves_for_Segmentation/data/Prostate/test.csv', type=str, required = False,
                        help='test data csv file')
    parser.add_argument('--binarise', type=str_to_bool, default=False,
                        help='Choose whether to binarise the segmentation map')
    parser.add_argument('--image_format', default='nifti',choices=['nifti', 'png'], type=str, 
                        help='state if image in nifti or png format')
    parser.add_argument('--labels', "--list", nargs="+",default=['Whole Prostate'],  
                        help='Label of classes. If not binarise, Prostate:["TZ", "PZ"], Chest: ["L-lung", "R-lung"], Abdomen: ["spleen",  "rkid", "lkid",  "gall", "eso", "liver", "sto",  "aorta",  "IVC",  "veins", "pancreas",  "rad", "lad"]')
    
    # Training parameters
    parser.add_argument('--classes', type=int, default=3, metavar='LR',
                        help='number of classes. 14 for abdomen CT, 3 for Prostate MRI, 3 for chest x-ray.')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001,
                        help='weight decay')
    parser.add_argument('--equivariant_loss', default='cosine', type=str, choices=['MSE', 'cosine'],
                        help='If you chose a model which forces equivariance with a contrastive loss, which loss function do you choose?')
    parser.add_argument('--val_epoch', type=int, default=5,
                        help='The number of epochs after which to evaluate the validation set')
    parser.add_argument('--sliding_inference', type=str_to_bool, default=False,
                        help='Choose whether to perform sliding window inference. Only required for abdomen')
    parser.add_argument('--beta1', type=float, default=0.0, help='Adam beta param (default: 0.0)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta param (default: 0.999)')
    
    #GPU usage
    parser.add_argument('--device', type=str, default='cuda', help='Which device the training is on')
    parser.add_argument('--gpus', type=int, default=1, 
                        help='number oF GPUs')
    parser.add_argument('--nodes', type=int, default=1,
                        help='number of nodes')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether to use deterministic training')

    #Output directory
    parser.add_argument('--output_dir', default='.../Sheaves_for_Segmentation/data/Prostate/output_images', type=str, required = False, 
                        help='output directory to save images and results')
    return parser

def main(args):
    pl.seed_everything(args.seed, workers=True)
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    data = ProstateMRIDataModule(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            label = True,
            image = True,
            binarise = args.binarise,
            contrastive = args.contrastive,
            image_size = args.image_size,
            csv_train_img=args.training_data,
            csv_val_img=args.validation_data,
            csv_test_img=args.test_data,
        )    
    if args.modeltype == 'ShapeVQUnet':
        model = ShapeVQUNet(args)
    elif args.modeltype == 'HybridShapeVQUnet':
        model = ShapeVQUNetHybrid(args)
    elif args.modeltype == 'HybridSE3VQUnet':
        model = VQHybridSECNN(args)
    elif args.modeltype == '3DSE3VQUnet':
        model = VQSE3UNET(args)
    else:
        model = None
     
    torch.cuda.set_device(0)
    model = model.to(args.device)

    os.makedirs(args.output_dir, exist_ok = True)
    if args.contrastive == True:
        trainimg = TrainD4(args, model = model)
    else:
        trainimg = TrainSE3(args, model = model)

    trainimg.train(train_dataset = data.train_dataloader(),  val_dataset = data.val_dataloader())  
    
    trainimg.test(model = model, model_path = os.path.join(args.output_directory, "checkpoints/bestsheafmodel.pt"), test_dataset = data.test_dataloader())    


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Vector quantisation for segmentation training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.quantise = 'spatial'
    args.out_ch = 2 if args.binarise else args.classes
    

    assert args.out_ch == args.classes
    assert args.topo_epoch < args.epochs
    
    if args.contrastive == True:
        assert args.modeltype in  ['ShapeVQUnet', 'HybridShapeVQUnet']
        del args.multiplicity, args.num_features, args.se3channel, args.group, args.multiplicity3D
    if args.dataset == 'chest' or 'prostate':
        args.sliding_inference == False 
    if args.modeltype != 'HybridShapeVQUnet': 
        del args.ch_3D
    if args.modeltype in ['ShapeVQUnet', '3DSE3VQUnet']: 
       for i in range(len(args.image_size)):
          try:
             assert args.image_size[i]%(2**(len(args.dec_blocks))) == 0, f'Dimension.{i} of your image input is not divisible by number of levels in your module to produce integer dimension features' 
          except AssertionError as msg:
             raise(AssertionError(msg))      

    main(args)