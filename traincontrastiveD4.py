
import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils as vutils
from loss import DiceLoss, FocalLoss
from einops import rearrange
import nibabel as nib
from collections import defaultdict
import torchio as tio
import torch.distributed as dist
import torchvision
from topoloss import *
import cripser as cr
import gudhi,gudhi.hera,gudhi.wasserstein
from monai.metrics import DiceMetric, compute_meandice, compute_hausdorff_distance, compute_average_surface_distance



def weights_init(m):
    classname = m.__class__.__name__
    if  isinstance(m,  nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif  isinstance(m,  nn.Conv3d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif  isinstance(m,  nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif  isinstance(m,  nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class TrainD4:
    def __init__(self, args, model):
        self._model = model#.to(device = device)
        self.device = args.device
        self.lr = args.lr
        self.wd = args.weight_decay
        self.image_format = args.image_format
        self.output_dir = args.output_dir
        self.equivariant_loss = args.equivariant_loss
        self.num_classes = args.classes
        self.patch_size = args.patch_size
        self.epochs = args.epochs
        self.topo_epoch = args.topo_epoch
        self.dim = args.dim
        self.compute_ph = args.compute_ph
        self.image_size = args.image_size
        self.topo_fg_only = args.topo_fg_only
        self.opt_vq = self.configure_optimizers(args.beta1, args.beta2)
        self.prepare_training(args.output_dir)
        

    def configure_optimizers(self, beta1, beta2):
        lr = self.lr
        opt_vq = torch.optim.AdamW(
            self._model.parameters(),
            lr=lr, eps=1e-08, betas=(beta1, beta2) )

        return opt_vq


    def save_images(self, outputs, imagespath, image_format, epoch, batch_idx, image_type):
        for i in range(outputs.shape[0]):
            output = torch.squeeze(outputs[i])
            directory = os.path.join(imagespath,'image{img}batch{batch}'.format(img = i,batch=batch_idx))
            os.makedirs(directory, exist_ok = True)
            filename =  os.path.join(directory,'{image_t}epoch{epoch}.{filetype}'.format(image_t=image_type,epoch=epoch, filetype = 'nii'))# if self.image_format == 'nifti' else 'png'))
            if image_format == 'nifti':
               output = output.cpu().detach().numpy().astype(np.float32)
               affine = np.eye(4)
               output = nib.Nifti1Image(output, affine)
               nib.save(output, filename)
            else:
               torchvision.utils.save_image(output, filename) 
                
        return None
    @staticmethod
    def prepare_training(output_dir):
        os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok = True)
    
    def _all_reduce(self, x, wr):
        # average x over all GPUs
        dist.all_reduce(x, dist.ReduceOp.SUM)
        return x / wr
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def divide_tensor_into_patches(self,tensor, patch_size):
        assert len(tensor.shape) == 2, "Input tensor should have 3 dimensions: (C, H, W)"
        H, W = tensor.shape
        assert H % patch_size == 0 and W % patch_size == 0, "Image dimensions should be divisible by patch_size"

        # Reshape the tensor to create patches
        patches = tensor.view(H // patch_size, patch_size, W // patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3)
        patches = patches.reshape(-1, patch_size, patch_size)

        return patches

    def divide_tensor_into_patches3D(self,tensor, patch_size):
        assert len(tensor.shape) == 3, "Input tensor should have 3 dimensions: (C, H, W)"
        H, W, Z = tensor.shape
        assert H % patch_size[0] == 0 and W % patch_size[1] == 0 and Z % patch_size[2] == 0, "Image dimensions should be divisible by patch_size"

        # Reshape the tensor to create patches
        patches = tensor.view(H // patch_size[0], patch_size[0], W // patch_size[1], patch_size[1],  Z // patch_size[2], patch_size[2])
        patches = patches.permute(0, 2,4, 1, 3,5)
        patches = patches.reshape(-1, patch_size[0], patch_size[1], patch_size[2])

        return patches

    def topo_labels2D(self,label, rows, cols, compute_ph):

        assert len(label.shape) == 3, "Input patches should have 4 dimensions: (N, C, patch_H, patch_W)"
        N, patch_H, patch_W = label.shape
        assert N == rows * cols, "Number of patches should be equal to rows * cols"

        # Initialize the empty image tensor
        
        label = label.to(device=self.device,  dtype=torch.float)
        label2 = torch.zeros(rows * patch_H, cols * patch_W)
        lab = {}
        for i in range(2 * rows * cols):
            lab[(i,)] = ()
 
        # Place patches sequentially in the image
        for i in range(rows):
          for j in range(cols):
            patch_idx = i * rows + j
            detached_patch = label[patch_idx]
            label2[i * patch_H:(i + 1) * patch_H, j * patch_W:(j + 1) * patch_W] = label[patch_idx] 
            if compute_ph == 'manual':
                lab[patch_idx,] = (1,0) if torch.sum(detached_patch) > 0 else (0,0)
                lab[(patch_idx)+(rows*cols),] = (1,0) if torch.sum(label2) > 0 else (0,0)
            else:
                pd = cr.computePH(detached_patch.cpu())
                lab[patch_idx,] = tuple([np.sum(pd[:,0] == i) for i in range(2)])
                pd2 = cr.computePH(label2)
                lab[(patch_idx)+(rows*cols),] = tuple([np.sum(pd[:,0] == i) for i in range(2)])
        return lab
    
    def topo_labels3D(self, label, rows, cols, dep, compute_ph):
 
        
        N, patch_H, patch_W, patch_Z = label.shape
        assert N == rows * cols * dep, "Number of patches should be equal to rows * cols * dep"

        # Initialize the empty image tensor
        
        label2 = torch.zeros(rows * patch_H, cols * patch_W, dep * patch_Z)
        lab = {}
        for i in range(2 * rows * cols * dep):
            lab[(i,)] = ()
 
        # Place patches sequentially in the image
        for i in range(rows):
          for j in range(cols):
            for l in range(dep):
                patch_idx = i*((rows)**2) + j*cols + l if dep > 1 else i * rows + j 
                detached_patch = label[patch_idx]
                label2[i * patch_H:(i + 1) * patch_H, j * patch_W:(j + 1) * patch_W,  l * patch_Z:(l + 1) * patch_Z] = label[patch_idx]
                if compute_ph == 'manual':
                    lab[patch_idx,] = (1,0) if torch.sum(detached_patch) > 0 else (0,0)
                    lab[(patch_idx)+(rows*cols*dep),] = (1,0) if torch.sum(label2) > 0 else (0,0)
                else:
                    pd = cr.computePH(detached_patch.cpu())
                    lab[patch_idx,] = tuple([np.sum(pd[:,0] == i) for i in range(2)])
                    pd2 = cr.computePH(label2)
                    lab[(patch_idx)+(rows*cols*dep),] = tuple([np.sum(pd2[:,0] == i) for i in range(2)])
                
        return lab

    def Cellular_patches2D(self, patches, rows, cols):
      
        assert len(patches.shape) == 3, "Input patches should have 3 dimensions: (N, C, patch_H, patch_W)"
        N, patch_H, patch_W = patches.shape
        assert N == rows * cols, "Number of patches should be equal to rows * cols"

        cells = [patches[x] for x in range(patches.shape[0])]


        # Initialize the empty image tensor
      
        image = torch.zeros(rows * patch_H, cols * patch_W)
        


        # Place patches sequentially in the image
        for i in range(rows):
          for j in range(cols):
               
               patch_idx = i + j*cols 

               image[i * patch_H:(i + 1) * patch_H, j * patch_W:(j + 1) * patch_W] = patches[patch_idx]
               cells.append(image)
   

        return cells

    def Cellular_patches3D(self, patches, rows, cols, dep):
      
        assert len(patches.shape) == 4, "Input patches should have 4 dimensions: (N, C, patch_H, patch_W)"
        N, patch_H, patch_W, patch_Z = patches.shape
        assert N == rows * cols * dep, "Number of patches should be equal to rows * cols"

        cells = [patches[x] for x in range(patches.shape[0])]


        # Initialize the empty image tensor
      
        image = torch.zeros(rows * patch_H, cols * patch_W, dep * patch_Z)
        
        # Place patches sequentially in the image
        for i in range(rows):
          for j in range(cols):
            for l in range(dep):
               
               patch_idx = i*((rows)**2) + j*cols + l if dep > 1 else i * rows + j

               image[i * patch_H:(i + 1) * patch_H, j * patch_W:(j + 1) * patch_W,  l * patch_Z:(l + 1) * patch_Z] = patches[patch_idx]
               cells.append(image)
   

        return cells
    
     
    def train(self, train_dataset, val_dataset):
        steps_per_epoch = len(train_dataset)
        min_loss = np.inf
        min_dice = np.inf
        writer = SummaryWriter()
        dice_loss = DiceLoss(self.num_classes)
        patch_dim = [int(self.image_size[i]/self.patch_size[i]) for i in range(len(self.image_size))]

        for epoch in range(self.epochs):
            with tqdm(range(len(train_dataset))) as pbar:
                for k , imgs in zip(pbar, train_dataset):
                    topoloss = 0
                    vqloss = 0
                    equivariantloss = 0
                    invariantloss = 0
                    diceloss = 0
                    t2img, t2lab, adcimage = imgs['t2image'].to(device=self.device,  dtype=torch.float), imgs['t2labels'].to(device=self.device,  dtype=torch.long), imgs['adcimage'].to(device=self.device,  dtype=torch.float)
                    img = torch.cat((t2img, adcimage), dim = 1)
                    lab = t2lab.as_tensor()
                    lab = torch.squeeze(self._one_hot_encoder(lab), dim = 2)
                    labt = lab[:,1:] if self.topo_fg_only == False else torch.unsqueeze(torch.sum(lab[:,1:],dim=1), dim =1)
                    out0, latentt20, latentadc0, vqt20, q_loss0 = self._model(img)
                    out0 = torch.softmax(out0, dim = 1)
                    out0t = out0[:,1:] if self.topo_fg_only == False else torch.unsqueeze(torch.sum(out0[:,1:],dim=1), dim =1)
                    vqloss += q_loss0
                    diceloss += dice_loss(out0, lab, softmax=False)

                    if epoch > self.topo_epoch:
                       for i in range(labt.shape[0]):
                            for j in range(labt.shape[1]):
                                patchesl = self.divide_tensor_into_patches3D(labt[i,j], self.patch_size) if self.dim == '3D' else self.divide_tensor_into_patches2D(labt[i,j], self.patch_size) 
                                patch_lab = self.topo_labels3D(patchesl, patch_dim[0], patch_dim[1], patch_dim[2], self.compute_ph) if self.dim == '3D' else self.topo_labels2D(patchesl, patch_dim[0], patch_dim[1], self.compute_ph)
                                patcheso = self.divide_tensor_into_patches3D(out0t[i,j], self.patch_size) if self.dim == '3D' else self.divide_tensor_into_patches2D(out0t[i,j], self.patch_size) 
                                patch_out = self.Cellular_patches3D(patcheso, patch_dim[0], patch_dim[1], patch_dim[2]) if self.dim == '3D' else self.Cellular_patches2D(patcheso, patch_dim[0], patch_dim[1])
                                topoloss += topo_loss(patch_out, patch_lab)
                    
                    for i in range(4):
                        for j in range(2):
                            if i+j > 0:
                               image = img.rot90(i, (2, 3)).rot90(j*2, (2, 4))
                               label = lab.rot90(i, (2, 3)).rot90(j*2, (2, 4))

                               out, latentt2, latentadc, vqt2, q_loss = self._model(image)
                               out = torch.softmax(out, dim = 1)
                               diceloss += dice_loss(out, label, softmax=False)
                               vqloss += q_loss
                               latentt2T = latentt2.rot90(-i, (2, 3)).rot90(-j*2, (2, 4))
                               latentadcT = latentadc.rot90(-i, (2, 3)).rot90(-j*2, (2, 4))
                               if self.equivariant_loss == 'MSE':
                                 equivariantloss += (torch.sum((latentt20 - latentt2T)**2) + torch.sum((latentadc0 - latentadcT)**2))/(latentadc0.shape[2]*latentadc0.shape[3]*latentadc0.shape[4]) 
                                 invariantloss += (torch.sum((latentt2 - latentadc)**2))/(latentadc.shape[2]*latentadc.shape[3]*latentadc.shape[4]) 
                               else:
                                 equivariantloss += torch.exp(-1*torch.sum(torch.sum(torch.reshape(latentt20, (latentt20.shape[0]* latentt20.shape[1], -1)) * torch.reshape(latentt2T, (latentt20.shape[0]* latentt20.shape[1], -1)), dim = 1) / (latentt20.shape[2]*latentt20.shape[3]*latentt20.shape[4]))) 
                                 equivariantloss += torch.exp(-1*torch.sum(torch.sum(torch.reshape(latentadc0, (latentadc0.shape[0]* latentadc0.shape[1], -1)) * torch.reshape(latentadcT, (latentadc0.shape[0]* latentadc0.shape[1], -1)), dim = 1) / (latentadc0.shape[2]*latentadc0.shape[3]*latentadc0.shape[4]))) 
                                 invariantloss += torch.exp(-1*torch.sum(torch.sum(torch.reshape(latentt2, (latentt2.shape[0]* latentt2.shape[1], -1)) * torch.reshape(latentadc, (latentadc.shape[0]* latentadc.shape[1], -1)), dim = 1) / (latentadc.shape[2]*latentadc.shape[3]*latentadc.shape[4]))) 
                            else:
                                continue
                                                
                    total_loss = diceloss/8+ vqloss/8 + topoloss/100 + equivariantloss + invariantloss
                    print('epoch:%.3f'% (epoch),'topo_loss: %.3f' % (topoloss/100), 'seg: %.3f' % (diceloss/8), 'qloss: %.3f' % (vqloss/8), 'equivariantloss: %.3f' % (equivariantloss), 'invariantloss: %.3f' % (invariantloss))
                    writer.add_scalar('VQLoss/train', total_loss,  k*epoch)
                    writer.add_scalar('seg_loss/train', diceloss,  k*epoch)
                    self.opt_vq.zero_grad()
                    total_loss.backward()
                    self.opt_vq.step()
         
                    pbar.set_postfix(
                        total_Loss=np.round(total_loss.cpu().detach().numpy().item(), 5),
                        dice=np.round(diceloss.cpu().detach().numpy().item(), 3)
                    )
                    pbar.update(0)
                checkpoint_path = os.path.join(self.output_dir, "checkpoints")
                val_dice, val_loss = self.val(val_dataset, epoch)
                torch.save(self._model.state_dict(), os.path.join(checkpoint_path, "bestsheafmodel.pt")) if val_loss < min_loss  else None
                min_loss = val_loss if val_loss < min_loss else min_loss
                min_dice = val_dice if val_dice < min_dice else min_dice
                print('min_val_dice:%.3f'% (min_dice))
    def val(self,  val_dataset, epoch):
        loss = defaultdict(list)        
        dice_loss = DiceLoss(self.num_classes-1)
        writer = SummaryWriter()

        with tqdm(range(len(val_dataset))) as pbar:
                for k, imgs in enumerate(val_dataset):
                    img, lab = imgs['t2image'].to(device=self.device,  dtype=torch.float), imgs['t2labels'].to(device=self.device,  dtype=torch.long)
                    lab = torch.squeeze(self._one_hot_encoder(lab), dim = 2)

                   
                    with torch.no_grad():
                        out, latentt2, latentadc, vqt2, q_loss = self._model(img)
                        outv = torch.softmax(out, dim = 1)
                        outh = torch.where(outv>0.5, 1, 0)
                        dice = dice_loss(outv[:,1:], lab[:,1:], softmax = False)
                        hd = compute_hausdorff_distance(outh,lab, include_background=False, percentile = 95)
                        val_loss = dice + q_loss
                    loss['dice'].append(dice)
                    loss['val_loss'].append(val_loss) 
                    print('epoch:%.3f'% (epoch), 'val q Loss: %.3f' % (q_loss), 'val dice Loss 1: %.3f' % (dice), 'val hd Loss: %.3f' % (torch.mean(hd)))
                    writer.add_scalar('VQLoss/val',val_loss,  k*epoch)
                    writer.add_scalar('seg_loss/val',dice,  k*epoch)
                    image_path = os.path.join(self.output_dir, "sheafval")
                    out = torch.softmax(outv, dim=1)
                    out = torch.argmax(out, dim = 1)
                    lab = torch.argmax(out, dim = 1)
                    self.save_images(out, image_path,  self.image_format,epoch, k, 'seg_image')
                    self.save_images(img, image_path,  self.image_format, epoch, k, 'original_image')
                    self.save_images(lab, image_path,  self.image_format, epoch, k, 'label_image')
        dice_mean = torch.mean(torch.stack(loss['dice'])) if len(loss['dice']) > 1 else loss['dice'][0]
        val_loss_mean = torch.mean(torch.stack(loss['val_loss'])) if len(loss['val_loss']) > 1 else loss['val_loss'][0]
        return dice_mean.cpu().detach().numpy(), val_loss_mean.cpu().detach().numpy()   
    
    def test(self, model,  model_path, test_dataset):
         _model =  model.to(device=self.device)
         _model.load_state_dict(torch.load(model_path))
         writer = SummaryWriter()
         dice_loss = DiceLoss(self.num_classes-1)
         loss = defaultdict(list)

         with tqdm(range(len(test_dataset))) as pbar:
                for k, imgs in enumerate(test_dataset):
                    img, lab = imgs['t2image'].to(device=self.device,  dtype=torch.float), imgs['t2labels'].to(device=self.device,  dtype=torch.long) 
                    lab = torch.squeeze(self._one_hot_encoder(lab), dim = 2)
                    with torch.no_grad():
                        out, latentt2, latentadc, vqt2, q_loss  = self._model(img)
                        outv = torch.softmax(out, dim = 1)
                        outh = torch.where(outv>0.5, 1, 0)
                        dice = dice_loss(outv[:,1:], lab[:,1:], softmax = False)
                        hd = compute_hausdorff_distance(outh,lab, include_background=False, percentile = 95)
                        test_loss = dice + q_loss
                    loss['dice'].append(dice)
                    loss['test_loss'].append(test_loss) 
                    print('test sample:%.3f'% (k), 'test q Loss: %.3f' % (q_loss), 'test dice Loss 1: %.3f' % (dice), 'test hd Loss 1: %.3f' % (torch.mean(hd)))
                    writer.add_scalar('VQLoss/test',test_loss,  k)
                    writer.add_scalar('seg_loss/test',dice,  k)
                    image_path = os.path.join(self.output_dir, "sheaftest")
                    out = torch.softmax(outv, dim=1)
                    out = torch.argmax(out, dim = 1)
                    lab = torch.argmax(out, dim = 1)
                    self.save_images(out, image_path,  self.image_format, self.epochs, k, 'seg_image')
                    self.save_images(img, image_path,  self.image_format, self.epochs, k, 'original_image')
                    self.save_images(lab, image_path,  self.image_format, self.epochs, k, 'label_image')
         dice_mean = torch.mean(torch.stack(loss['dice'])) if len(loss['dice']) > 1 else loss['dice'][0]
         test_loss_mean = torch.mean(torch.stack(loss['test_loss'])) if len(loss['test_loss']) > 1 else loss['test_loss'][0]

         return dice_mean.cpu().detach().numpy(), test_loss_mean.cpu().detach().numpy()
