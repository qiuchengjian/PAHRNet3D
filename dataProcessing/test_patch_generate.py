import numpy as np
import SimpleITK as sitk
import torch

import parameters as para


def test_seg_generate(ct_arr, device, net, val=True):
    origin_shape = ct_arr.shape
    count = np.zeros(origin_shape, dtype=np.int16)
    probability_map = np.zeros(origin_shape, dtype=np.float32)
    if val:
        step_x, step_y, step_z = para.val_stride_x, para.val_stride_y, para.val_stride_z
    else:
        step_x, step_y, step_z = para.stride_x, para.stride_y, para.stride_z
    x_shape, y_shape, z_shape = para.patch_size[0], para.patch_size[1], para.patch_size[2]
    threshold = para.threshold
    x_start_max = origin_shape[0] - x_shape
    y_start_max = origin_shape[1] - y_shape
    z_start_max = origin_shape[2] - z_shape
    for x_start in range(0, x_start_max + 1, step_x):
        for y_start in range(0, y_start_max + 1, step_y):
            for z_start in range(0, z_start_max + 1, step_z):
                ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start+x_shape, y_start:y_start+y_shape,
                                              z_start:z_start+z_shape]).to(device)
                ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
                outputs = net(ct_tensor)
                count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
                probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
                    += np.squeeze(outputs.cpu().detach().numpy())
                del outputs

    x_start = x_start_max
    for y_start in range(0, y_start_max+1, step_y):
        for z_start in range(0, z_start_max+1, step_z):
            ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start + x_shape, y_start:y_start + y_shape,
                                          z_start:z_start + z_shape]).to(device)
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
            probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
                += np.squeeze(outputs.cpu().detach().numpy())
            del outputs
    x_start = x_start_max
    y_start = y_start_max
    for z_start in range(0, z_start_max+1, step_z):
        ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start + x_shape, y_start:y_start + y_shape,
                                      z_start:z_start + z_shape]).to(device)
        ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        outputs = net(ct_tensor)
        count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
        probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
            += np.squeeze(outputs.cpu().detach().numpy())
        del outputs
    x_start = x_start_max
    z_start = z_start_max
    for y_start in range(0, y_start_max+1, step_y):
        ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start + x_shape, y_start:y_start + y_shape,
                                      z_start:z_start + z_shape]).to(device)
        ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        outputs = net(ct_tensor)
        count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
        probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
            += np.squeeze(outputs.cpu().detach().numpy())
        del outputs
    x_start = x_start_max
    y_start = y_start_max
    z_start = z_start_max
    ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start + x_shape, y_start:y_start + y_shape,
                                  z_start:z_start + z_shape]).to(device)
    ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
    outputs = net(ct_tensor)
    count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
    probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
        += np.squeeze(outputs.cpu().detach().numpy())
    del outputs

    z_start = z_start_max
    for x_start in range(0, x_start_max + 1, step_x):
        for y_start in range(0, y_start_max + 1, step_y):
            ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start + x_shape, y_start:y_start + y_shape,
                                          z_start:z_start + z_shape]).to(device)
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
            probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
                += np.squeeze(outputs.cpu().detach().numpy())
            del outputs
    z_start = z_start_max
    y_start = y_start_max
    for x_start in range(0, x_start_max + 1, step_x):
        ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start + x_shape, y_start:y_start + y_shape,
                                      z_start:z_start + z_shape]).to(device)
        ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
        outputs = net(ct_tensor)
        count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
        probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
            += np.squeeze(outputs.cpu().detach().numpy())
        del outputs

    y_start = y_start_max
    for x_start in range(0, x_start_max + 1, step_x):
        for z_start in range(0, z_start_max + 1, step_z):
            ct_tensor = torch.FloatTensor(ct_arr[x_start:x_start + x_shape, y_start:y_start + y_shape,
                                          z_start:z_start + z_shape]).to(device)
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)
            count[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] += 1
            probability_map[x_start: x_start + x_shape, y_start: y_start + y_shape, z_start: z_start + z_shape] \
                += np.squeeze(outputs.cpu().detach().numpy())
            del outputs

    pred_seg = np.zeros_like(probability_map)
    pred_seg[probability_map >= (threshold * count)] = 1
    print('maxOverlapCount:', np.max(count), 'minOverlapCount:', np.min(count))
    return pred_seg
