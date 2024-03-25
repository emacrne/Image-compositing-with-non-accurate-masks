import torch
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from torchvision import transforms
from PIL import Image
from torch import nn
import random
import math


def test(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          sched,
          device: torch.device,
          epoch,
          psnr_list,
          rmse_list,
          model_name):  # -> Tuple[Dict[str, List], List]:
    
    results = {
               "test_loss": []
               }
        
    #model1.eval()
    loss_fn.eval()   #####!!!
    test_loss, psnr = 0, 0
    rmse = 0
    crit2 = nn.L1Loss()
    mse_loss = nn.MSELoss()

    transform2pil = transforms.ToPILImage()

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
            
            gt_pred = model(img, bg, mask, train=False)
            #gt_pred = model(X)
           
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


            res.save("dipl/results"+ model_name + "/results" + str(batch) + ".png", "PNG")

            dif = torch.logical_xor(mask, real_mask)
            area = (torch.zeros(dif.shape)).to(device)
            area[dif] = 1
            mse = mse_loss(gt_pred*area, gt*area) * 0.8 + mse_loss(gt_pred*(1-area), gt*(1-area)) * 0.2

            loss = loss_fn(gt_pred, gt) * 0.1
            loss += mse 
            #loss = loss_fn(gt_pred.cpu().detach(), gt.cpu().detach())  #.cpu().detach()
            test_loss += loss.item()

            #dif = torch.logical_xor(mask, real_mask)
            #area = (torch.zeros(dif.shape)).to(device)
            #area[dif] = 1
            mse = mse_loss(gt_pred*area, gt*area) * 0.5 + mse_loss(gt_pred*(1-area), gt*(1-area)) * 0.5
            if mse != 0:
                psnr +=  20 * math.log10(1.0) - 10 * math.log10(mse.item())
                rmse_t = torch.sqrt(mse)
                rmse += rmse_t.item()

       
        test_loss = test_loss / len(dataloader)
        psnr = psnr / len(dataloader)
        psnr_list[epoch+1] = psnr
        rmse = rmse / len(dataloader)
        rmse_list[epoch+1] = rmse
        #print(psnr_list)
        sched.step(test_loss)
        torch.cuda.empty_cache()

        curr_lr = optimizer.param_groups[0]['lr']

        print(
            f"test_loss: {test_loss:.4f} | "
            f"psnr: {psnr:.4f} | "
            f"rmse: {rmse:.4f} | "
            f"LR:{curr_lr:.4f}"
        )

        results["test_loss"].append(test_loss)
        
    return test_loss, psnr_list, rmse_list