import torch
import torchvision

para = torch.load('./outputs/imagenet_baseline_resnet152/last_epoch_ckpt.pth.tar',map_location='cpu')['model']
newmodel = {}
for k, v in para.items():
    if not k.startswith("module.teacher_encoder."):
        continue
    if 'fc' in k:
        continue
    old_k = k
    k = k.replace("module.teacher_encoder.", "")

    print(old_k, "->", k)
    newmodel[k] = v
import pickle as pkl
res = {"model": newmodel, "__author__": "WENHAO", "matching_heuristics": True}
with open('/dev/shm/unsupervised_pretrained_byol_152.pkl', "wb") as f:
        pkl.dump(res, f)