import os
import sys
import multiprocessing as mp

BIAS = 0
RANK_SIZE = 8
RANK_TABLE_FILE = "/home/cvgroup/wcr/mindcv_latest/hccl.json"


def f(rank_id, script, args):
    os.environ["RANK_ID"] = f"{rank_id}"  # logical id
    os.environ["DEVICE_ID"] = f"{rank_id + BIAS}"  # physical id
    os.environ["RANK_TABLE_FILE"] = RANK_TABLE_FILE
    print(f"Launching rank: {os.getenv('RANK_ID')}, device: {os.getenv('DEVICE_ID')}, pid: {os.getpid()}")
    os.system(f"python -u {script} {args}")


if __name__ == '__main__':
    mp.set_start_method("spawn")

    script_, args_ = sys.argv[1], ' '.join(sys.argv[2:])
    print(f"Script: {script_}, Args: {args_}")
    processes = [mp.Process(target=f, args=(i, script_, args_)) for i in range(RANK_SIZE)]
    [p.start() for p in processes]
    [p.join() for p in processes]

# python launch_dist.py train_with_func.py --config configs/resnet/resnet_50_ascend.yaml --data_dir /ms_test/ImageNet_Original --val_while_train True
