from dv import AedatFile
import os
import json
import numpy as np

def get_items(path, interv = 40000):
    frame_exposure_time = []
    frame_interval_time = []
    with AedatFile(path) as f:
        #print(f.names)
        # extract timestamps of each frame 
        size = f['events'].size
        events = np.hstack([packet for packet in f['events'].numpy()])
        frame_exposure_time = [[frame.timestamp_start_of_exposure, frame.timestamp_end_of_exposure] for frame in f['frames']]
        curr = frame_exposure_time[0][0]
        while(curr + interv < frame_exposure_time[-1][1]):
            frame_interval_time.append([curr,    curr + interv])       ## [1607928583387944, 1607928583410285]
            curr = curr + interv
    with AedatFile(path) as f:
        frames = [packet.image for packet in f['frames']]
    return size, events, frame_exposure_time, frame_interval_time, frames

def get_files(base_dir="./data"):
    files = {}
    for f in os.listdir(base_dir):
        if f.endswith(".aedat4"):
            aedat4_f = os.path.join(base_dir, f)
            files[f] = {"path":os.path.join(base_dir, f),
                "aedat4":aedat4_f,
                "name":f.split(".")[0]
            }
    return files

def get_annotations(base_dir="./data"):
    bb_gt = {}
    for file in os.listdir(base_dir):
        if(file.endswith(".txt")):
            f_path = os.path.join(base_dir,file)
            if(os.path.isfile(f_path)):
                with open(f_path,'r') as f:
                    content = [[float(x) for x in line.split(",")] for line in f.read().strip().split("\n")]
                    bb_gt[file.split(".")[0]] = content
    return bb_gt

def get_dataset_info(base_dir="./data"):
    with open(f'{base_dir}/info.json', "r") as read_file:
        data = json.load(read_file)
    return data