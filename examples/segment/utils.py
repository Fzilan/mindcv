import numpy as np
from mindspore import Tensor
import mindspore.common.dtype as mstype
import cv2

def get_net_output(network, input, flip=True):
    """
    infer image

    Args:
        network (Cell): the infer network.
        input (np.ndarray): a batch of pre-processsed image in shape of (N, C, H, W)
        flip(boolean): add result of flipped images
    Returns:
        output(np.ndarray): predict prob, in shape of (N, C, H, W)

    """    
    output = network(Tensor(input, mstype.float32))
    output = output.asnumpy()

    if flip:
        input_filp = input[:, :, :, ::-1]
        output_flip = network(Tensor(input_filp, mstype.float32))
        output += output_flip.asnumpy()[:, :, :, ::-1]

    return output

def resize_to_origin(input, origin_size, batch_size):
    """
    resize outputs to original size to match labels
    Args:
        input (np.ndarray): a batch of images in shape of (N, C, H, W)
        origin_size(list): image original size to be resized   
        batch_size: batch size
    Returns:
        output_list(list): in length of batch_size, instance in shape of (H, W, C)

    """
    output_list = []
    for idx in range(batch_size):
        # ?
        res = input[idx][:,:,:].transpose((1, 2, 0))
        h, w = origin_size[idx][0], origin_size[idx][1]
        res = cv2.resize(res, (w, h))
        output_list.append(res)
    return output_list


def calculate_hist(flattened_label, flattened_pred, num_classes):
    k = (flattened_label>=0) & (flattened_label < num_classes)
    res = np.bincount(num_classes * flattened_label[k].astype(np.int32) + flattened_pred[k], 
                      minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return res
            
def calculate_batch_hist(preds:list, labels:list, batch_size:int, num_classes:int):
    """
    Args: 
        preds(list):
        labels(list):
    Returns:
        output_list(list): in length of batch_size, instance in shape of (H, W, C)

    """ 
    batch_hist = np.zeros((num_classes, num_classes))
    for idx in range(batch_size):
        pred = preds[idx].flatten()
        gt = labels[idx].flatten()
        batch_hist += calculate_hist(pred, gt, num_classes)
    return batch_hist

def apply_eval(eval_param_dict):
    net = eval_param_dict["net"]
    net.set_train(False)
    ds = eval_param_dict["dataset"]    
    args = eval_param_dict['args']

    hist = np.zeros((args.num_classes, args.num_classes))

    batch_idx = 0
    inner_batch_idx = 0
    # NCHW
    batch_image = np.zeros((args.batch_size, 3, args.crop_size, args.crop_size), dtype=np.float32)
    batch_label = []
    batch_origin_size = []

    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):
        img_np = data["data"]
        label_np = data["label"]
        
        batch_image[inner_batch_idx] = img_np
        batch_label.append(cv2.imdecode(np.frombuffer(label_np, dtype=np.uint8), cv2.IMREAD_GRAYSCALE))
        batch_origin_size.append(label_np.shape)
        inner_batch_idx += 1

        if inner_batch_idx == args.batch_size:
            batch_idx += 1
            batch_output = get_net_output(network=net,
                                          input=batch_image, 
                                          flip=args.flip)
            
            batch_output = resize_to_origin(input=batch_output, 
                                            origin_size=batch_origin_size,
                                            batch_size=args.batch_size)
            
            hist += calculate_batch_hist(preds=batch_output, 
                                         labels=batch_label,
                                         batch_size=args.batch_size,
                                         num_classes=args.num_classes)

            inner_batch_idx = 0
            batch_image = np.zeros((args.batch_size, 3, args.crop_size, args.crop_size), 
                                   dtype=np.float32)
            batch_origin_size = []
            batch_label = []
            
            print('processed {} images'.format(batch_idx*args.batch_size))

    IoU = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    mIoU = np.nanmean(IoU)

    return IoU, mIoU




                    

