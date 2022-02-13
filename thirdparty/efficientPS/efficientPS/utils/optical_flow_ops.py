import math
import torch
from torch import nn
import torch.nn.functional as F


def flow_warp(x, flow, mul=True):
    """
    Warp the current image using the flow so that it maps to the previous image
    x: [B, C, H, W] (curr_image)
    flow: [B, 2, H, W] (flow)
    """
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1).to(x.device)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W).to(x.device)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), dim=1).float()

    vgrid = grid + flow

    # Scale the grid to [-1, 1]
    vgrid = torch.stack([2.0 * vgrid[:, 0, :, :] / max(W-1, 1) - 1.0, 2.0 * vgrid[:, 1, :, :] / max(H-1, 1) - 1.0],
                        dim=1)
    vgrid = vgrid.permute(0, 2, 3, 1)
    output = F.grid_sample(x, vgrid, padding_mode="border", align_corners=True)
    mask = torch.ones(x.size(), device=x.device)
    mask = F.grid_sample(mask, vgrid, padding_mode="zeros", align_corners=True)

    mask[mask < 0.9999] = 0
    mask[mask > 0] = 1

    if mul:
        return output * mask
    else:
        return output, mask


def density2vector(prob_map, normalise=True):
    """
    Convert the probability map into the flow vector
    prob_map: The generated probability map
    normalise: if True, normalise using Softmax
    """
    flow = prob2flow(prob_map, normalise)
    return flow


def vector2density(flow, support_size):
    return flow2Distribution(flow, support_size)


def flow2Distribution(flow, support_size):
    B, _, H, W = flow.size()
    flow = torch.clamp(flow, min=-support_size, max=support_size)
    x = flow[:, 0, :, :]
    y = flow[:, 1, :, :]
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0_safe = torch.clamp(x0, min=-support_size, max=support_size)
    y0_safe = torch.clamp(y0, min=-support_size, max=support_size)
    x1_safe = torch.clamp(x1, min=-support_size, max=support_size)
    y1_safe = torch.clamp(y1, min=-support_size, max=support_size)

    wt_x0 = (x1 - x) * torch.eq(x0, x0_safe).float()
    wt_x1 = (x - x0) * torch.eq(x1, x1_safe).float()
    wt_y0 = (y1 - y) * torch.eq(y0, y0_safe).float()
    wt_y1 = (y - y0) * torch.eq(y1, y1_safe).float()

    wt_tl = wt_x0 * wt_y0
    wt_tr = wt_x1 * wt_y0
    wt_bl = wt_x0 * wt_y1
    wt_br = wt_x1 * wt_y1

    mask_tl = torch.eq(x0, x0_safe).float() * torch.eq(y0, y0_safe).float()
    mask_tr = torch.eq(x1, x1_safe).float() * torch.eq(y0, y0_safe).float()
    mask_bl = torch.eq(x0, x0_safe).float() * torch.eq(y1, y1_safe).float()
    mask_br = torch.eq(x1, x1_safe).float() * torch.eq(y1, y1_safe).float()

    wt_tl *= mask_tl
    wt_tr *= mask_tr
    wt_bl *= mask_bl
    wt_br *= mask_br

    out = torch.zeros((B, (2 * support_size + 1)**2, H, W), device=flow.device)
    label_tl = (y0_safe + support_size) * (2 * support_size + 1) + x0_safe + support_size
    label_tr = (y0_safe + support_size) * (2 * support_size + 1) + x1_safe + support_size
    label_bl = (y1_safe + support_size) * (2 * support_size + 1) + x0_safe + support_size
    label_br = (y1_safe + support_size) * (2 * support_size + 1) + x1_safe + support_size

    out.scatter_add_(1, label_tl.unsqueeze(1).long(), wt_tl.unsqueeze(1))
    out.scatter_add_(1, label_tr.unsqueeze(1).long(), wt_tr.unsqueeze(1))
    out.scatter_add_(1, label_bl.unsqueeze(1).long(), wt_bl.unsqueeze(1))
    out.scatter_add_(1, label_br.unsqueeze(1).long(), wt_br.unsqueeze(1))

    return out


def prob2flow(prob_map, normalise):
    corner_prob, corner_flow = prob2cornerFlow(prob_map, normalise)
    out = cornerFlow2Expectation(corner_prob, corner_flow)
    return out

def prob2cornerFlow(prob_map, normalise=True):
    def indice2Flow(ind, d):
        # ind: [B, 1, H, W]
        return torch.cat([ind % d - d // 2, ind // d - d // 2], 1)

    if normalise:
        normaliser = nn.Softmax(dim=1)
        prob_map = normaliser(prob_map)

    B, C, H, W = prob_map.size()
    d = int(math.sqrt(C))
    pr = prob_map.reshape(B, d, d, -1).permute(0, 3, 1, 2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    max_pool = nn.MaxPool2d(kernel_size=d-1, stride=1, return_indices=True)
    out, indice = max_pool(avg_pool(pr))
    indice += indice // (d-1)  # In original coordinates
    indice = indice.squeeze().reshape(B, H, W).unsqueeze(1)
    lt_prob = torch.gather(prob_map, 1, indice)
    lt_flow = indice2Flow(indice, d).float()
    rt_prob = torch.gather(prob_map, 1, indice+1)
    rt_flow = indice2Flow(indice+1, d).float()
    lb_prob = torch.gather(prob_map, 1, indice+d)
    lb_flow = indice2Flow(indice+d, d).float()
    rb_prob = torch.gather(prob_map, 1, indice+d+1)
    rb_flow = indice2Flow(indice+d+1, d).float()

    return [lt_prob, rt_prob, lb_prob, rb_prob], [lt_flow, rt_flow, lb_flow, rb_flow]


def cornerFlow2Expectation(corner_prob, corner_flow):
    corner_prob_sum = sum(corner_prob)
    corner_prob_n = [prob/corner_prob_sum for prob in corner_prob]
    out = torch.cat([corner_flow[1][:, 0, :, :].unsqueeze(1) - corner_prob_n[0] - corner_prob_n[2],
                     corner_flow[2][:, 1, :, :].unsqueeze(1) - corner_prob_n[0] - corner_prob_n[1]],
                    dim=1)
    return out


def downsample_flow(flow, scale_factor):
    assert scale_factor <= 1
    B, C, H, W = flow.size()
    if flow.size(1) == 2:
        # Dense format --> This is the common one I think
        flow = F.interpolate(flow, scale_factor=scale_factor, mode='bilinear', align_corners=True,recompute_scale_factor=True)
        mask = torch.ones((B, 1, int(H * scale_factor), int(W * scale_factor)), dtype=torch.float, device=flow.device)
    else:
        # Sparse format
        flow = F.avg_pool2d(flow, int(1/scale_factor))
        mask = (flow[:, 2, :, :].unsqueeze(1) > 0).float()
        flow = flow[:, :2, :, :] / (flow[:, 2, :, :].unsqueeze(1) + 1e-9)

    return flow, mask


def resize_dense_vector(vec, des_height, des_width):
    ratio_height = float(des_height / vec.size(2))
    ratio_width = float(des_width / vec.size(3))
    vec = F.interpolate(
        vec, (des_height, des_width), mode='bilinear', align_corners=True, recompute_scale_factor=True)
    if vec.size(1) == 1:
        vec = vec * ratio_width
    else:
        vec = torch.stack(
            [vec[:, 0, :, :] * ratio_width, vec[:, 1, :, :] * ratio_height],
            dim=1)
    return vec


# MFN Stuff #
def upsample_kernel2d(w, device):
    c = w // 2
    kernel = 1 - torch.abs(c - torch.arange(w, dtype=torch.float32, device=device)) / (c + 1)
    kernel = kernel.repeat(w).view(w, -1) * kernel.unsqueeze(1)
    return kernel.view(1, 1, w, w)


def Upsample(img, factor):
    if factor == 1:
        return img
    B, C, H, W = img.shape
    batch_img = img.view(B*C, 1, H, W)
    batch_img = F.pad(batch_img, [0, 1, 0, 1], mode='replicate')
    kernel = upsample_kernel2d(factor * 2 - 1, img.device)
    upsamp_img = F.conv_transpose2d(batch_img, kernel, stride=factor, padding=(factor-1))
    upsamp_img = upsamp_img[:, :, : -1, :-1]
    _, _, H_up, W_up = upsamp_img.shape
    return upsamp_img.view(B, C, H_up, W_up)

    # if factor == 1:
    #     return 1
    # upsampled_img = F.interpolate(img, scale_factor=factor, align_corners=True, mode="bilinear")
    # return upsampled_img
