import math
import torch

# im1_torch, im2_torch in shape (N, C, H, W)
def extract_flow_torch(model, im1_torch, im2_torch):
    # interpolate image, make new_H, mew_W divide by 64
    assert im1_torch.shape == im2_torch.shape
    N, C, H, W = im1_torch.shape
    device = im1_torch.device
    new_H = int(math.floor(math.ceil(H / 64.0) * 64.0))
    new_W = int(math.floor(math.ceil(W / 64.0) * 64.0))
    im1_torch = torch.nn.functional.interpolate(input=im1_torch, size=(new_H, new_W), mode='bilinear',
                                                 align_corners=False)
    im2_torch = torch.nn.functional.interpolate(input=im2_torch, size=(new_H, new_W), mode='bilinear',
                                                 align_corners=False)
    model.eval()
    with torch.no_grad():
        flo12 = model(im1_torch, im2_torch)
    flo12 = 20.0 * torch.nn.functional.interpolate(input=flo12, size=(H, W), mode='bilinear',
                                                          align_corners=False)
    flo12[:, 0, :, :] *= float(W) / float(new_W)
    flo12[:, 1, :, :] *= float(H) / float(new_H)
    return flo12

# im1_np, im2_np in shape (C, H, W)
def extract_flow_np(model, im1_np, im2_np):
    im1_torch = torch.from_numpy(im1_np).unsqueeze(0).to(torch.device('cuda'))
    im2_torch = torch.from_numpy(im2_np).unsqueeze(0).to(torch.device('cuda'))
    flo12_torch = extract_flow_torch(model, im1_torch, im2_torch)
    flo12_np = flo12_torch.detach().cpu().squeeze(0).numpy()
    return flo12_np

