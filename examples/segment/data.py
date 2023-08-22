from typing import List, Optional, Type, Union
import numpy as np
import cv2
import mindspore.dataset as de
cv2.setNumThreads(0)

class SegDataset:
    def __init__(self,
                 image_mean,
                 image_std,
                 data_file='',
                 batch_size=32,
                 crop_size=512,
                 max_scale=2.0,
                 min_scale=0.5,
                 ignore_label=255,
                 num_classes=21,
                 num_readers=2,
                 num_parallel_calls=4,
                 shard_id=None,
                 shard_num=None,
                 shuffle = True,
                 is_training=True):

        self.data_file = data_file
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.ignore_label = ignore_label
        self.num_classes = num_classes
        self.num_readers = num_readers
        self.num_parallel_calls = num_parallel_calls
        self.shard_id = shard_id
        self.shard_num = shard_num
        self.shuffle = shuffle
        self.is_training = is_training
        assert max_scale > min_scale

    def train_preprocess_(self, image, label):
        # bgr image
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)
        label_out = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        # random scaling
        sc = np.random.uniform(self.min_scale, self.max_scale)
        new_h, new_w = int(sc * image_out.shape[0]), int(sc * image_out.shape[1])
        image_out = cv2.resize(image_out, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        label_out = cv2.resize(label_out, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # mean std
        image_out = (image_out - self.image_mean) / self.image_std

        # pad or random crop 
        h_, w_ = max(new_h, self.crop_size), max(new_w, self.crop_size)
        pad_h, pad_w = h_ - new_h, w_ - new_w
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
            label_out = cv2.copyMakeBorder(label_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_label)
        offset_h = np.random.randint(0, h_ - self.crop_size + 1)
        offset_w = np.random.randint(0, w_ - self.crop_size + 1)
        image_out = image_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size, :]
        label_out = label_out[offset_h: offset_h + self.crop_size, offset_w: offset_w + self.crop_size]

        # random flip
        if np.random.uniform(0.0, 1.0) > 0.5:
            image_out = image_out[:, ::-1, :]
            label_out = label_out[:, ::-1]

        image_out = image_out.transpose((2, 0, 1))
        image_out = image_out.copy()
        label_out = label_out.copy()

        return image_out, label_out         

    def eval_preprocess_(self, image):
        image_out = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), cv2.IMREAD_COLOR)

        # propositionally resize, make the longer size equal to crop size
        h, w, _ = image_out.shape
        if h > w:
            new_h = self.crop_size
            new_w = int(1.0 * self.crop_size * w / h)
        else:
            new_w = self.crop_size
            new_h = int(1.0 * self.crop_size * h / w)
        image_out = cv2.resize(image_out, (new_w, new_h))

        # mean, std
        image_out = (image_out - self.image_mean) / self.image_std

        # pad to crop_size
        pad_h = self.crop_size - image_out.shape[0]
        pad_w = self.crop_size - image_out.shape[1]
        if pad_h > 0 or pad_w > 0:
            image_out = cv2.copyMakeBorder(image_out, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        # hwc to chw
        image_out = image_out.transpose((2, 0, 1))

        # return img
        # original_size = [[h, w]]
        # print(type(image_out),type(np.array(original_size)))
        # print(image_out.shape, np.array(original_size).shape)
        # print(image_out.shape)
        return image_out

    def get_dataset(self):

        # print("data.py::get_dataset ", self.data_file)
        data_set = de.MindDataset(dataset_files=self.data_file, 
                                  columns_list=["data", "label"],
                                  shuffle=True, 
                                  num_parallel_workers=self.num_readers,
                                  num_shards=self.shard_num, 
                                  shard_id=self.shard_id)
        # print_dataset(data_set)
        if self.is_training:
            input_columns= ["data", "label"]
            output_columns=["data", "label"]
            transforms_list = self.train_preprocess_
        
        else:
            input_columns= ["data"]
            output_columns=["data"]
            transforms_list = self.eval_preprocess_

        data_set = data_set.map(operations=transforms_list,
                                input_columns=input_columns,
                                output_columns=output_columns,
                                # input_columns=["data", "label"],
                                # output_columns=["data", "label"],
                                num_parallel_workers=self.num_parallel_calls)
        # print_dataset(data_set) 
        if self.is_training and self.shuffle:
            data_set = data_set.shuffle(buffer_size=self.batch_size * 10)
            data_set = data_set.batch(self.batch_size, drop_remainder=True)
        # print_dataset(data_set)
        # data_set = data_set.repeat(repeat)
        return data_set

def create_segment_dataset(
        name,
        data_dir,
        image_mean,
        image_std,
        shuffle,
        batch_size=32,
        crop_size=512,
        max_scale=2.0,
        min_scale=0.5,
        ignore_label=225,
        num_classes=21,
        num_parallel_workers=8,
        # num_readers,
        # num_parallel_calls,
        shard_id=0,
        shard_num=1,
        is_training=True,
):
    if name == "voc" or name == "vocaug":
        dataset = SegDataset(
            image_mean=image_mean,
            image_std=image_std,
            data_file=data_dir,
            batch_size=batch_size,
            crop_size=crop_size,
            max_scale=max_scale,
            min_scale=min_scale,
            ignore_label=ignore_label,
            num_classes=num_classes,
            num_readers=num_parallel_workers,
            num_parallel_calls=num_parallel_workers,
            shard_id=shard_id,
            shard_num=shard_num,
            shuffle=shuffle,
            is_training=is_training          
        )

        return dataset.get_dataset()

    else:
        raise NotImplementedError

def print_dataset(data_set):
    count = 0
    for item in data_set.create_dict_iterator(output_numpy=True):
        print("sample: {}".format(item))
        count += 1
    print("Got {} samples".format(count))


# # test
# ds = SegDataset(image_mean=[103.53, 116.28, 123.675],image_std=[57.375, 57.120, 58.395]
#                 ,data_file="/Users/fanzhilan/Documents/HUAWEI/Requirement/Dataset/sample.mindrecord",
#                 batch_size=2,is_training=False)
# ds = ds.get_dataset()
# for data in ds.create_dict_iterator(output_numpy=True):
#     print(data["label"].shape)
#     label = data["label"]
#     label1 = cv2.imdecode(np.frombuffer(label, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
#     label2 = label.asnumpy()
#     print(label1.shape,label2.shape)
    # break

#     print(data.keys())
#     print(data["data"].shape, data["label"].shape)
    # print(data["data"].shape, data["original_size"].shape)
# print(ds.num_classes)
