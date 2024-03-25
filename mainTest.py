import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, io
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import cv2
from torchinfo import summary
from tqdm.auto import tqdm
from timeit import default_timer as timer
import torchvision.models as models
import convAEk5 as model_type
import math
import vgg_loss
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from rembg import remove
from skimage import io
from pytorch_msssim import ssim, ms_ssim


torch.cuda.empty_cache()

class ImageFolderCustom(Dataset):

    def __init__(self, img_dir: str, bg_dir: str, train: bool, d_mask_dir:str) -> None:

        self.img_path = sorted(list(Path(img_dir).glob("*/*.jpg")))
        self.bg_path = sorted(list(Path(bg_dir).glob("*/*.jpg")) )
        self.deformed_mask_path = sorted(list(Path(d_mask_dir).glob("*.jpg")))
        self.mask_path = sorted(list(Path(img_dir).glob("*/*.png")))
        self.train = train

        n = 100

        self.img_path =  self.img_path[:n]
        self.mask_path =  self.mask_path[:n]
        self.deformed_mask_path =  self.deformed_mask_path[:n]    
        self.bg_path = self.bg_path[:n]  

        self.random_samples_idx = random.sample(range(len(self.img_path)), k=int(len(self.img_path)/2))

         

        """self.addColor = A.Compose([A.ColorJitter(always_apply=False, p=0.5)])
        self.addTransform = A.Compose(
            [
                #A.VerticalFlip(p=0.5),
                A.HorizontalFlip(p=0.5),
                #A.RandomSizedCrop((230, 255), 255, 255),
            ],
            additional_targets={
                #"img": "image",
                #"bg": "image",
                "mask": "image",
                "deformed_mask": "image"
            }
        )"""

        self.toTensor = A.Compose([
            ToTensorV2()
        ])
        

    def load_image(self, index: int, path: str) -> Image.Image:
        image_path = path[index]
        return Image.open(image_path)


    def deform_mask(self, mask: Image.Image):

        #deforming masks with morphology 

        size = 255
        mask = mask.resize((size, size))
        mask = np.array(mask)
        
        mask[mask > 0] = 255
        kernel = np.ones((20, 20))
        blurred = cv2.GaussianBlur(mask, (5, 5), cv2.BORDER_DEFAULT)
        blurred[blurred > 0] = 255
        deform_mask = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        deform_mask = cv2.morphologyEx(deform_mask, cv2.MORPH_CLOSE, kernel)

        """
        # deforming mask with scaling to 1/8 of size

        w, h = mask.shape
        scaling_factor = 1/8
        smaller = cv2.resize(mask, (0, 0), fx=scaling_factor, fy=scaling_factor)
        restored = cv2.resize(smaller, (h, w))

        (T, thresh) = cv2.threshold(restored, 200, 255, cv2.THRESH_BINARY)
        restored[restored > 0] = 255"""

        return deform_mask
    

    def deform_mask2(self, mask: Image.Image, img):

        # second method of deforming masks with findCountours

        size =255
        mask = mask.resize((size, size))
        img = img.resize((size, size))
        mask_np = np.array(mask)
        img = np.array(img)
        blurred = cv2.GaussianBlur(mask_np, (5, 5), cv2.BORDER_DEFAULT)
        blurred[blurred > 0] = 255
        #deform_mask = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        #deform_mask = cv2.morphologyEx(deform_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        deform_mask = np.zeros_like(mask)

        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.drawContours(deform_mask, [approx], -1, (255, 255, 255), thickness=cv2.FILLED)
            
        #deformed_mask = Image.fromarray(deform_mask)
        #masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)

        return deform_mask


    def addHoles(self, deform_mask):
        # generate and draw random holes

        num_holes = 3
        min_radius = 1
        max_radius = 1

        for _ in range(num_holes):
            radius = np.random.randint(min_radius, max_radius + 1)
            center_x = np.random.randint(0, 255)
            center_y = np.random.randint(0, 255)
            radius_perturbation = np.random.randint(-10, 11)
            center_x_perturbation = np.random.randint(-20, 21)
            center_y_perturbation = np.random.randint(-20, 21)
            
            noisy_radius = max(radius + radius_perturbation, 0)  # non-negative radius
            noisy_center_x = max(center_x + center_x_perturbation, 0)
            noisy_center_y = max(center_y + center_y_perturbation, 0)
            
            cv2.circle(deform_mask, (noisy_center_x, noisy_center_y), noisy_radius, 0, -1)

        return deform_mask


    def __len__(self) -> int:
        return len(self.img_path) 


    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        img = self.load_image(index, self.img_path)
        bg = self.load_image(index, self.bg_path)
        mask = (self.load_image(index, self.mask_path)).convert("L")
        #deformed_mask = self.load_image(index, self.deformed_mask_path)
        #deformed_mask = remove(img).convert("L")


        # randomly select masks for first and second deformation
        if index in self.random_samples_idx:
            deformed_mask = self.deform_mask(mask)
        else:
            deformed_mask = self.deform_mask2(mask, img)
        
        #deformed_mask = self.deform_mask(mask)
        #deformed_mask = self.addHoles(deformed_mask)

        size = 255

        image = np.array(img.resize((size, size)))
        bg = np.array(bg.resize((size, size)))
        mask = np.array(mask.resize((size, size)))
        #deformed_mask = np.array(deformed_mask.resize((size, size)))
        #deformed_mask[deformed_mask > 0] = 255

        transformed_img = self.toTensor(image=image)
        img = transformed_img["image"] /255
        transformed_bg = self.toTensor(image=bg)
        bg = transformed_bg["image"] /255
        transformed_mask = self.toTensor(image=mask)
        mask = transformed_mask["image"] /255
        transformed_dmask = self.toTensor(image=deformed_mask)
        deformed_mask = transformed_dmask["image"] /255
    
        gt = bg * (1 - mask) + img * mask

        return img, bg, deformed_mask, gt, mask
    

if __name__ == '__main__':

    #test_path = Path("dipl/AIM-500")
    test_path = Path("dipl/test2")
    bg_path = Path("dipl/background_dataset")
    d_mask_path = Path("dipl/segmentation_masks2")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
  
    test_data = ImageFolderCustom(img_dir=test_path, bg_dir=bg_path, train=False, d_mask_dir=d_mask_path)

    # display_random_images(test_data, n=5, seed=None)

    ###########################################
    #here choose what model type you want to use
    model_name = "K5"
    ##############################################

    NUM_WORKERS = os.cpu_count()
    # print(f"batch size: {BATCH_SIZE} & workers: {NUM_WORKERS} ")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    NUM_EPOCHS = 1


    dataloader = DataLoader(dataset=test_data,
                                        num_workers=NUM_WORKERS,
                                        shuffle=False)

    model1 = model_type.ConvAutoencoder()
    model1.load_state_dict(torch.load(r"/home/ema/results/model" + model_name + "a.pth")) #, map_location=torch.device('cpu')
    model2 = model_type.ConvAutoencoder()
    model2.load_state_dict(torch.load(r"/home/ema/results/model" + model_name + "b.pth")) #, map_location=torch.device('cpu')
    model1 = model1.to(device)
    model2 = model2.to(device)
    fusion_model = model_type.CustomFusionModel(model1, model2).to(device)

    loss_fn = vgg_loss.VGGLoss().to(device)

    #loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=fusion_model.parameters(), lr=0.002, weight_decay=1e-5)  

    loss_fn.eval()   #####!!!
    test_loss, psnr = 0, 0
    rmse = 0
    sim = 0
    crit2 = nn.L1Loss()
    mse_loss = nn.MSELoss()

    transform2pil = transforms.ToPILImage()

    start_time = timer()

    with torch.no_grad():
        for batch, (img, bg, mask, gt, real_mask) in enumerate(dataloader):

            #img, bg, mask, gt = addTransform([img, bg, mask, gt])

            img = img.to(device)
            bg = bg.to(device)
            mask = mask.to(device)
            gt = gt.to(device)
            real_mask = real_mask.to(device)

            size = img.shape[3]
            #print(size)

            res = Image.new("RGB", (3*size, 2*size), "white")

            combined = img * mask + bg * (mask -1)*(-1)

            a = transform2pil(combined[0])
            #a.save("dipl/results/a" + "_" + str(batch) + ".png", "PNG")

            res.paste(a, (0, 0))
            
            #X = torch.cat([img, bg, mask], dim=1)
            
            #gt_pred.requires_grad_(True)
            
            gt_pred = fusion_model(img, bg, mask, train=False)
            #gt_pred = model1(X)
           
            res_pil = transform2pil(gt_pred[0])
            res.paste(res_pil, (size, 0))
            gt_pil = transform2pil(gt[0])
            res.paste(gt_pil, (2*size, 0))
            real_pil = transform2pil(real_mask[0])
            mask_pil = transform2pil(mask[0])
            res.paste(mask_pil, (0, size))
            res.paste(real_pil, (size, size))
            img_pil = transform2pil(img[0])
            res.paste(img_pil, (2*size, size))


            res.save("dipl/tests/test" + model_name + "/results" + str(batch) + ".png", "PNG")

            """dif = torch.logical_xor(mask, real_mask)                # za utezevanje psnr preverjanja
            area = (torch.zeros(dif.shape)).to(device)
            area[dif] = 1
            mse = mse_loss(gt_pred*area, gt*area) *0.5 + mse_loss(gt_pred*(1-area), gt*(1-area)) *0.5"""
            wrong = torch.zeros(gt.shape).to(device)

            mse = mse_loss(gt_pred, gt)

            gt_pred = (gt_pred + 1) / 2
            gt = (gt + 1) / 2

            ms_ssim_val = ms_ssim( gt_pred, gt, data_range=1, size_average=False )


            if mse != 0:    
                psnr +=  20 * math.log10(1.0) - 10 * math.log10(mse.item())
                rmse_t = torch.sqrt(mse)
                rmse += rmse_t.item()
                sim += ms_ssim_val.item()

        end_time = timer()
        time = end_time - start_time
        print(f"Total training time: {time:.3f} seconds")

        fps = 100/time

        print(f"fps: {fps:.3f}")

        test_loss = test_loss / len(dataloader)
        psnr = psnr / len(dataloader)
        rmse = rmse / len(dataloader)
        sim = sim / len(dataloader)
        #print(psnr_list)
        torch.cuda.empty_cache()

        print(
            f"model: {model_name} |"
            f"psnr: {psnr:.4f} | "
            f"rmse: {rmse:.6f} | "
            f"ms_sim: {sim:.6f} | "
        )

    