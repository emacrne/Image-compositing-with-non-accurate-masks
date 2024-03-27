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
import train
import test
import utils
import CAE6 as model_type
import vgg_loss
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

torch.cuda.empty_cache()


class ImageFolderCustom(Dataset):

    def __init__(self, img_dir: str, bg_dir: str, train: bool) -> None:

        self.img_path = sorted(list(Path(img_dir).glob("*/*.jpg")))
        self.bg_path = list(Path(bg_dir).glob("*/*.jpg")) 
        random.shuffle(self.bg_path)                     # if i want backgrounds to shuffled randomly
        self.mask_path = sorted(list(Path(img_dir).glob("*/*.png")))
        self.train = train
        num_train = 10000
        num_test = 50
        if self.train == True:  #for training it uses first 1000 of images 
            self.img_path =  self.img_path[:num_train]
            self.mask_path =  self.mask_path[:num_train]
            self.bg_path =  self.bg_path[:len(self.img_path)]
        else:   #for testing we use last 50 of images
            self.img_path =  self.img_path[-num_test:]
            self.mask_path =  self.mask_path[-num_test:]      
            self.bg_path = self.bg_path[-len(self.img_path):]  

        self.random_samples_idx = random.sample(range(len(self.img_path)), k=int(len(self.img_path)/2))

        self.addColor = A.Compose([A.ColorJitter(always_apply=False, p=0.1)])
        self.addTransform = A.Compose(
            [
                #A.VerticalFlip(p=0.05),
                A.HorizontalFlip(p=0.5),
                #A.RandomSizedCrop((230, 255), 255, 255),
            ],
            additional_targets={
                #"img": "image",
                #"bg": "image",
                "mask": "image",
                "deformed_mask": "image"
            }
        )

        self.toTensor = A.Compose([
            ToTensorV2()
        ])
        

    def load_image(self, index: int, path: str) -> Image.Image:
        image_path = path[index]
        return Image.open(image_path)
    

    def deform_mask(self, mask: Image.Image):   #morphological deformations
        mask = mask.resize((size, size))
        mask_np = np.array(mask)
        kernel = np.ones((20, 20))
        blurred = cv2.GaussianBlur(mask_np, (5, 5), cv2.BORDER_DEFAULT)
        blurred[blurred > 0] = 255
        deform_mask = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
        deform_mask = cv2.morphologyEx(deform_mask, cv2.MORPH_CLOSE, kernel)
        #kernel = np.ones((20, 20))
        #deform_mask = cv2.dilate(deform_mask, kernel)
        #deformed_mask = Image.fromarray(deform_mask)

        return deform_mask


    def deform_mask2(self, mask: Image.Image, img):     # fingContourus deformations
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
            
            noisy_radius = max(radius + radius_perturbation, 0)  
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

        if index in self.random_samples_idx:
            deformed_mask = self.deform_mask(mask)
        else:
            deformed_mask = self.deform_mask2(mask, img)
        
        deformed_mask = self.addHoles(deformed_mask)

        size = 255
        
        image = np.array(img.resize((size, size)))
        bg = np.array(bg.resize((size, size)))
        mask = np.array(mask.resize((size, size)))
        #deformed_mask = np.array(deformed_mask.resize((size, size)))

        if self.train:
            """transformed_img = self.addColor(image=image)
            image = transformed_img["image"]
            transformed_bg = self.addColor(image=bg)
            bg = transformed_bg["image"]"""
            
            transformed = self.addTransform(image=image, mask=mask, deformed_mask=deformed_mask)
            transformed_bg = self.addTransform(image=bg)

            image = transformed["image"]
            bg = transformed_bg["image"]
            mask = transformed["mask"]
            deformed_mask = transformed["deformed_mask"]


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

    train_path = Path("dipl/train2/")
    test_path = Path("dipl/test2/")
    bg_path = Path("dipl/background_dataset")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    size = 255
  
    train_data = ImageFolderCustom(img_dir=train_path, bg_dir=bg_path, train=True)
    test_data = ImageFolderCustom(img_dir=test_path, bg_dir=bg_path, train=False)

    #utils.display_random_images(test_data, n=5, seed=None)


    model_name = "K5"
    BATCH_SIZE = 5
    NUM_WORKERS = os.cpu_count()
    #print(f"batch size: {BATCH_SIZE} & workers: {NUM_WORKERS} ")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    NUM_EPOCHS = 50

    train_dataloader = DataLoader(dataset=train_data,
                                         batch_size=BATCH_SIZE,
                                         num_workers=NUM_WORKERS,
                                         shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                        num_workers=NUM_WORKERS,
                                        shuffle=False)
    
    model1 = model_type.ConvAutoencoder().to(device)
    model2 = model_type.ConvAutoencoder().to(device)
    fusion_model = model_type.CustomFusionModel(model1, model2).to(device)


    """loss_fn = vgg_loss.WeightedLoss([vgg_loss.VGGLoss(),
                                  nn.MSELoss(),
                                  vgg_loss.TVLoss(p=1)],
                                 [1, 10, 0]).to(device)"""
    
    loss_fn = vgg_loss.VGGLoss().to(device)
    #loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=fusion_model.parameters(), lr=0.002, weight_decay=1e-5)  #spremenjen learaning rate(1e-4) && , weight_decay=0.0001
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)
    start_time = timer()

    model_train_results, psnr_list, rmse_list= train.train(model=fusion_model,
                            dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            sched=sched,
                            epochs=NUM_EPOCHS,
                            device=device,
                            model_name=model_name)
    
    # model.load_state_dict(torch.load())
    """model_test_results, psnr = test.test(model=model,
                            dataloader=test_dataloader,
                            loss_fn=loss_fn,
                            optimizer=optimizer,
                            sched=sched,
                            device=device)"""


    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")

    utils.plot_loss_curves(model_train_results, model_name=model_name)

    print(psnr_list)
    print(rmse_list)

    max_psnr = max(psnr_list.values())  
    max_psnr_indx = [k for k, v in psnr_list.items() if v == max_psnr] 

    max_rmse = max(rmse_list.values())  
    max_rmse_indx = [k for k, v in rmse_list.items() if v == max_rmse] 

    print(max_psnr, max_psnr_indx)
    print(max_rmse, max_rmse_indx)

    utils.save_model(model=model1, target_dir="results/", model_name="model" + model_name + "a.pth")
    utils.save_model(model=model2, target_dir="results/", model_name="model" + model_name + "b.pth")


    