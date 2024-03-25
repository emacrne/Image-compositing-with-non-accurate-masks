import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchvision import transforms
import test
from torch import nn
import random
import albumentations as A

def train(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          test_dataloader:  torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          sched,
          epochs: int,
          device: torch.device,
          model_name):  # -> Tuple[Dict[str, List], List]:
    
    results = {"train_loss": [],
               "test_loss": [],
               }
    
    psnr_list = {-1: 0}
    rmse_list = {-1:0}

    for epoch in tqdm(range(epochs)):
        #model.train()
        loss_fn.train()
        train_loss, train_acc = 0, 0
        transform2pil = transforms.ToPILImage()
       
        normalize = transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                #max_pixel_value=255.0,
            )
        normalize2 = transforms.Compose([
            transforms.Normalize(
                mean=[0.0],
                std=[1.0],
                #max_pixel_value=255.0,
            )
        ])

        crit2 = nn.L1Loss()
        mse_loss = nn.MSELoss()

        for batch, (img, bg, mask, gt, real_mask) in enumerate(dataloader):

            img = img.to(device)
            bg = bg.to(device)
            mask = mask.to(device)
            gt = gt.to(device)
            real_mask = real_mask.to(device)

            optimizer.zero_grad()
            combined = img * mask + bg * (1 - mask)

            if batch < 10:
                cp_pil = transform2pil(combined[0])
                cp_pil.save("dipl/results" + model_name+ "/train" + str(batch) + ".png", "PNG")

            #X = torch.cat([img, bg, mask], dim=1)

            #gt_pred = model(X)

            gt_pred = model(img, bg, mask, train=True)
            #print(img_features.shape, bg_features.shape)
           
            #gt_act = loss_fn.get_features(gt)

            #gt_pred.requires_grad_(True)"""

            dif = torch.logical_xor(mask, real_mask)
            area = (torch.zeros(dif.shape)).to(device)
            area[dif] = 1
            mse = mse_loss(gt_pred*area, gt*area) * 0.8 + mse_loss(gt_pred*(1-area), gt*(1-area)) * 0.2

            loss = loss_fn(gt_pred, gt) * 0.1
            loss += mse 
            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(dataloader)
        #sched.step()

        #curr_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            #f"LR:{curr_lr:.4f}"
        )

        results["train_loss"].append(train_loss)
        # results["train_acc"].append(train_acc)

        #if epoch % 5 == 0:
        model_test_results, psnr_list, rmse_list = test.test(model=model,
                        dataloader=test_dataloader,
                        loss_fn=loss_fn,
                        optimizer=optimizer,
                        sched=sched,
                        device=device,
                        epoch=epoch,
                        psnr_list=psnr_list,
                        rmse_list=rmse_list,
                        model_name=model_name)
        
        results["test_loss"].append(model_test_results)

    return results, psnr_list, rmse_list