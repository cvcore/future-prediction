import cv2
import numpy as np
import flow_vis
import torch
import torch.nn.functional as F
import wandb
from torch import align_tensors
from efficientPS.utils.sequence import pad_packed_images

from efficientPS.utils.optical_flow_ops import *


def visualiseFlow(img_pair, flow_gt, flow_pred, flow_prob, ds, idx=0):
    dim = flow_pred[0].size(1)
    H, W = img_pair[0].size()[2:]
    with torch.no_grad():
        full_vis_list = []
        for idx in range(img_pair[0].size(0)):
            raw_img0 = recoverImage(img_pair[0][idx].data)
            raw_img1 = recoverImage(img_pair[1][idx].data)
            raw_img0 = raw_img0 * 255.0
            raw_img1 = raw_img1 * 255.0

            for l in range(len(flow_pred)):
                # Image

                # for i in range(3):
                #     raw_img0[i, :, :] = raw_img0[i, :, :] * 255.0 / torch.max(raw_img0[i, :, :])
                #     raw_img1[i, :, :] = raw_img1[i, :, :] * 255.0 / torch.max(raw_img1[i, :, :])

                vis_list = [raw_img0, raw_img1]

                # Ground truth flow
                gt_flow, valid_mask = downsample_flow(flow_gt, 1. / 2 ** (ds - l))
                gt_flow = F.interpolate(gt_flow, (H, W), mode="nearest", recompute_scale_factor=True)[idx]
                valid_mask = F.interpolate(valid_mask, (H, W), mode="nearest", recompute_scale_factor=True)[idx]

                max_mag1 = torch.max(torch.norm(gt_flow, 2, 0))

                # predicted flow
                pred_flow = flow_pred[l]
                pred_flow = F.interpolate(pred_flow, (H, W), mode='nearest', recompute_scale_factor=True)[idx]
                max_mag2 = torch.max(torch.norm(pred_flow, 2, 0))

                max_mag = max(float(max_mag1), float(max_mag2))
                # print("GT Flow", gt_flow.size())
                # print("GT Flow", gt_flow.shape)
                gt_flow_np = gt_flow.detach().cpu().permute(1, 2, 0).numpy()
                pred_flow_np = pred_flow.cpu().permute(1, 2, 0).numpy()
                gt_flow_vis = flow_vis.flow_to_color(gt_flow_np, convert_to_bgr=False)
                pred_flow_vis = flow_vis.flow_to_color(pred_flow_np, convert_to_bgr=False)
                vis_list.append(torch.from_numpy(gt_flow_vis).permute(2, 0, 1).cuda())
                vis_list.append(torch.from_numpy(pred_flow_vis).permute(2, 0, 1).cuda())

                # epe error visualization
                epe_error = torch.norm(pred_flow - gt_flow, 2, 0, keepdim=False) * valid_mask[0, :, :]
                normalizer = max(torch.max(epe_error), 1)
                epe_error = 1 - epe_error / normalizer
                vis_list.append(visualiseHeatmap(epe_error))

                # confidence map visualization
                prob = flow_prob[l].data
                prob = probGather(prob, normalize=True)
                if prob.size(2) != H or prob.size(3) != W:
                    prob = F.interpolate(prob, (H, W), mode='nearest', recompute_scale_factor=True)
                vis_list.append(visualiseHeatmap(prob[idx].squeeze(), cv2.COLORMAP_BONE))

                vis = torch.cat(vis_list, dim=2)
                if l == 0:
                    ms_vis = vis
                else:
                    ms_vis = torch.cat([ms_vis, vis], dim=1)

            full_vis_list.append(ms_vis.unsqueeze(0))
        return full_vis_list


def visualiseFlowMFN(img_pair, flow_gt, flow_pred, mask, dataset):
    # H, W = img_pair[0].size()[2:]
    with torch.no_grad():
        full_vis_list = []
        for idx in range(img_pair[0].size(0)):
            raw_img0 = recoverImage(img_pair[0][idx].data)
            raw_img1 = recoverImage(img_pair[1][idx].data)
            raw_img0 = raw_img0 * 255
            raw_img1 = raw_img1 * 255

            # Image
            vis_list = [raw_img0, raw_img1]

            # Mask
            valid_mask = mask[idx]
            # Ground truth flow
            gt_flow = flow_gt[idx]
            # predicted flow
            pred_flow = flow_pred[idx]

            # print(pred_flow.shape)
            gt_flow_np = gt_flow.detach().cpu().permute(1, 2, 0).numpy()
            pred_flow_np = pred_flow.cpu().permute(1, 2, 0).numpy()
            valid_mask_np = valid_mask.cpu().permute(1, 2, 0).numpy()

            # if dataset in ["Kitti15", "Kitti12"]:
                # gt_flow_np[:, :, 0] = (gt_flow_np[:, :, 0] * 64) + (2 ** 15)
                # gt_flow_np[:, :, 1] = (gt_flow_np[:, :, 1] * 64) + (2 ** 15)

            invalid_indices = np.argwhere(np.invert(valid_mask_np))
            gt_flow_np[invalid_indices[:, 0], invalid_indices[:, 1], :] = 0

            # print(np.min(gt_flow_np), np.max(gt_flow_np))

            # gt_flow_vis = flow_vis.flow_to_color(gt_flow_np, convert_to_bgr=False)
            # pred_flow_vis = flow_vis.flow_to_color(pred_flow_np, convert_to_bgr=False)
            gt_flow_vis = visualiseFlowRGB(gt_flow_np)
            pred_flow_vis = visualiseFlowRGB(pred_flow_np)
            vis_list.append(torch.from_numpy(gt_flow_vis).permute(2, 0, 1).type(torch.float).to(pred_flow.device))
            vis_list.append(torch.from_numpy(pred_flow_vis).permute(2, 0, 1).type(torch.float).to(pred_flow.device))

            vis = torch.cat(vis_list, dim=2)
            full_vis_list.append(vis.unsqueeze(0))

        return full_vis_list


def visualiseFlowRGB(flow):
    H, W, C = flow.shape[0], flow.shape[1], 3
    flow_hsv = np.zeros((H, W, C), dtype=np.uint8)
    flow_hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    flow_hsv[..., 0] = ang * 180 / np.pi / 2
    flow_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    flow_rgb = cv2.cvtColor(flow_hsv, cv2.COLOR_HSV2RGB)
    return flow_rgb


def recoverImage(img):
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device)
    img = img.permute(1, 2, 0) * std + mean
    return img.permute(2, 0, 1)


def visualiseHeatmap(x, method=cv2.COLORMAP_JET):
    x = np.uint8(x.cpu().numpy() * 255)
    x = torch.from_numpy(cv2.applyColorMap(x, method)).cuda()
    return x.permute(2, 0, 1).float()


def probGather(prob, normalize=True, return_indices=False):
    # gather probability for confidence map visualization
    # return shape: out, indice [B,1,H,W]
    if normalize:
        prob = F.softmax(prob, dim=1)
    B, C, H, W = prob.size()
    d = int(math.sqrt(C))
    pr = prob.reshape(B, d, d, -1).permute(0, 3, 1, 2)
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    max_pool = nn.MaxPool2d(kernel_size=d - 1, stride=1, return_indices=True)
    out, indice = max_pool(avg_pool(pr))
    out = 4 * out.squeeze().reshape(B, 1, H, W)
    if not return_indices:
        return out
    else:
        indice += indice / (d - 1)
        indice = indice.squeeze().reshape(B, 1, H, W)
        return out, indice


def visualiseDepth(img, depth_gt, depth_pred, ms_depth, min_depth, max_depth, dataset, iseg_boundary=None, iseg_object=None):
    with torch.no_grad():
        full_vis_list = []
        for idx in range(img.size(0)):
            raw_img0 = recoverImage(img[idx].data)
            raw_img0 = raw_img0 * 255.0

            # for i in range(3):
            #     raw_img0[i, :, :] = raw_img0[i, :, :] * 255.0

            # Image
            vis_list = [raw_img0]

            if dataset in ['Cityscapes', 'CityscapesSample', 'CityscapesDepth', 'CityscapesDepthSample', "KittiRawDepth", "KittiPanoptic"]:
                depth_pred_copy = depth_pred[idx].detach().clone()
                depth_pred_vis_mask = torch.logical_or(depth_pred_copy < 0.1, depth_pred_copy > 100.0)
                pred_depth_inv = torch.reciprocal(depth_pred_copy)
                pred_depth_inv[depth_pred_vis_mask] = 0
                pred_depth_vis = pred_depth_inv - torch.min(pred_depth_inv)
                pred_depth_vis = pred_depth_vis * (255. / torch.max(pred_depth_vis))
                vis_list.append(torch.cat([pred_depth_vis, pred_depth_vis, pred_depth_vis], dim=0))

                depth_gt_copy = depth_gt[idx].detach().clone()
                depth_gt_vis_mask = torch.logical_or(depth_gt_copy < 0.1, depth_gt_copy > 100.0)
                gt_depth_inv = torch.reciprocal(depth_gt_copy)
                gt_depth_inv[depth_gt_vis_mask] = 0
                gt_depth_vis = gt_depth_inv - torch.min(gt_depth_inv)
                gt_depth_vis = gt_depth_vis * (255. / torch.max(gt_depth_vis))
                vis_list.append(torch.cat([gt_depth_vis, gt_depth_vis, gt_depth_vis], dim=0))

                # Save the difference map as well
                pred_depth = depth_pred_copy
                pred_depth[depth_gt_vis_mask] = 0
                gt_depth = depth_gt_copy
                gt_depth[depth_gt_vis_mask] = 0
                diff_map = torch.abs(pred_depth - gt_depth)
                diff_map = diff_map - torch.min(diff_map)
                diff_map = diff_map * (255. / torch.max(diff_map))
                vis_list.append(torch.cat([diff_map, diff_map, diff_map], dim=0))

            else:
                pred_depth = (depth_pred[idx] - torch.min(depth_pred[idx])) * (255. / torch.max(depth_pred[idx]))
                vis_list.append(torch.cat([pred_depth, pred_depth, pred_depth], dim=0))

                gt_depth = (depth_gt[idx] - torch.min(depth_gt[idx])) * (255. / torch.max(depth_gt[idx]))
                vis_list.append(torch.cat([gt_depth, gt_depth, gt_depth], dim=0))

            # for ms in ms_depth:
            #     ms_up = F.interpolate(ms, scale_factor=4, mode="bilinear", align_corners=True, recompute_scale_factor=True)[0] * 255.0
            #     vis_list.append(torch.cat([ms_up, ms_up, ms_up], dim=0))

            # for vis in vis_list:
            #     print(vis.shape)

            if iseg_boundary is not None:
                iseg_boundary_copy = iseg_boundary[idx].detach().clone()
                iseg_boundary_copy = iseg_boundary_copy - torch.min(iseg_boundary_copy)
                iseg_boundary_copy = iseg_boundary_copy * (255. / torch.max(iseg_boundary_copy))
                vis_list.append(torch.cat([iseg_boundary_copy, iseg_boundary_copy, iseg_boundary_copy], dim=0))

            if iseg_object is not None:
                iseg_object_copy = iseg_object[idx].detach().clone()
                iseg_object_copy = iseg_object_copy - torch.min(iseg_object_copy)
                iseg_object_copy = iseg_object_copy * (255. / torch.max(iseg_object_copy))
                vis_list.append(torch.cat([iseg_object_copy, iseg_object_copy, iseg_object_copy], dim=0))

            vis = torch.cat(vis_list, dim=2)
            full_vis_list.append(vis.unsqueeze(0))
        return full_vis_list


def visualisePanopticDepth(img, depth_gt, depth_pred, class_pred=None, class_mask=None, dataset=None):
    with torch.no_grad():
        full_vis_list = []
        for idx in range(img.size(0)):
            raw_img0 = recoverImage(img[idx].data)
            raw_img0 = raw_img0 * 255.0

            # Image
            vis_list = [raw_img0]

            if dataset in ['Cityscapes', 'CityscapesSample', 'CityscapesDepth', 'CityscapesDepthSample', "KittiRawDepth", "CityscapesSeam", "KittiPanoptic"]:
                depth_pred_copy = depth_pred[idx].detach().clone()
                depth_pred_vis_mask = (depth_pred_copy < 0.1) | (depth_pred_copy > 100.0)
                pred_depth_inv = torch.reciprocal(depth_pred_copy)
                pred_depth_inv[depth_pred_vis_mask] = 0
                pred_depth_vis = pred_depth_inv - torch.min(pred_depth_inv)
                pred_depth_vis = pred_depth_vis * (255. / torch.max(pred_depth_vis))
                vis_list.append(torch.cat([pred_depth_vis, pred_depth_vis, pred_depth_vis], dim=0))

                depth_gt_copy = depth_gt[idx].detach().clone()
                depth_gt_vis_mask = (depth_gt_copy < 0.1) | (depth_gt_copy > 100.0)
                gt_depth_inv = torch.reciprocal(depth_gt_copy)
                gt_depth_inv[depth_gt_vis_mask] = 0
                gt_depth_vis = gt_depth_inv - torch.min(gt_depth_inv)
                gt_depth_vis = gt_depth_vis * (255. / torch.max(gt_depth_vis))
                vis_list.append(torch.cat([gt_depth_vis, gt_depth_vis, gt_depth_vis], dim=0))

                # Save the difference map as well
                pred_depth = depth_pred_copy
                pred_depth[depth_gt_vis_mask] = 0
                gt_depth = depth_gt_copy
                gt_depth[depth_gt_vis_mask] = 0
                diff_map = torch.abs(pred_depth - gt_depth)
                diff_map = diff_map - torch.min(diff_map)
                diff_map = diff_map * (255. / torch.max(diff_map))
                vis_list.append(torch.cat([diff_map, diff_map, diff_map], dim=0))

            else:
                pred_depth = (depth_pred[idx] - torch.min(depth_pred[idx])) * (255. / torch.max(depth_pred[idx]))
                vis_list.append(torch.cat([pred_depth, pred_depth, pred_depth], dim=0))

                gt_depth = (depth_gt[idx] - torch.min(depth_gt[idx])) * (255. / torch.max(depth_gt[idx]))
                vis_list.append(torch.cat([gt_depth, gt_depth, gt_depth], dim=0))

            if class_pred is not None:
                # Road, Buildings, Vehicles
                class_pred_copy = class_pred[idx].detach().clone()
                class_pred_vis_mask = (class_pred_copy < 0.1) | (class_pred_copy > 100.0)
                pred_class_inv = torch.reciprocal(class_pred_copy)
                pred_class_inv[class_pred_vis_mask] = 0

                # if class_mask is not None:
                #     class_mask_copy = class_mask[idx].detach().clone()
                #     pred_class_inv[~class_mask_copy] = 0

                class_pred_stuff = [pred_class_inv[0, :, :], pred_class_inv[1, :, :], pred_class_inv[2, :, :],
                                    pred_class_inv[3, :, :], pred_class_inv[4, :, :], pred_class_inv[5, :, :],
                                    pred_class_inv[6, :, :], pred_class_inv[7, :, :],]
                for stuff in class_pred_stuff:
                    stuff = stuff - torch.min(stuff)
                    stuff = stuff * (255. / torch.max(stuff))
                    stuff = stuff.unsqueeze(0)
                    vis_list.append(torch.cat([stuff, stuff, stuff], dim=0))

            # for vis in vis_list:
            #     print(vis.shape)

            # Resize the image by a factor of 0.25
            vis_list = [F.interpolate(v.unsqueeze(0), scale_factor=0.25, mode="bilinear", align_corners=True).squeeze(0) for v in vis_list]

            vis = torch.cat(vis_list, dim=2)
            full_vis_list.append(vis.unsqueeze(0))
        return full_vis_list


def visualiseSemanticSegmentation(img, sem_gt, sem_pred, dataset):
    sem_vis_list = []

    # sem_pred_small = [F.interpolate(s.unsqueeze(0).type(torch.FloatTensor), scale_factor=0.25, mode='nearest').type(torch.LongTensor).squeeze(0) for s in sem_pred]
    # sem_gt_small = [F.interpolate(s.unsqueeze(0).type(torch.FloatTensor), scale_factor=0.25, mode='nearest').type(torch.LongTensor).squeeze(0) for s in sem_gt]
    # img_small = [F.interpolate(s.unsqueeze(0), scale_factor=0.25, mode='bilinear', align_corners=True).squeeze(0) for s in img]

    for b in range(len(sem_gt)):
        if dataset in ['Cityscapes', "CityscapesSeam", "KittiPanoptic"]:
            class_labels = {0: "road", 1: "sidewalk", 2: "building" , 3: "wall", 4: "fence", 5: "pole", 6: "traffic_light",
                        7: "traffic_sign", 8: "vegetation", 9: "terrain", 10: "sky", 11: "person", 12: "rider",
                        13: "car", 14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle", 255: "others"}

        # sem_pred, _ = pad_packed_images(sem_pred)
        masks = {
            "predictions": {
                "mask_data": sem_pred[b].detach().cpu().numpy(),
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": sem_gt[b].detach().cpu().numpy(),
                "class_labels": class_labels
            }
        }
        sem_img = wandb.Image(img[b].permute(1, 2, 0).detach().cpu().numpy(), masks=masks)
        sem_vis_list.append(sem_img)

    return sem_vis_list
