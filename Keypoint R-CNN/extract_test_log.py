import os, sys, tqdm
import numpy as np
from bd_rate import bj_delta

h = w = 4

baseline = {
    'x264': {'bpp': np.array([0.9161, 0.5017, 0.2613, 0.1401]), 'bbox': [50.5213, 46.1293, 37.4434, 23.7803], 'keypoint': [59.4864, 52.6895, 41.2736, 24.1587]},
    'x265': {'bpp': np.array([1.0240, 0.5860, 0.3087, 0.1460]), 'bbox': [51.6328, 49.3236, 44.4877, 35.3849], 'keypoint': [61.2577, 57.8764, 50.2074, 37.7034]},
    'jpeg2000': {'bpp': np.array([1.1975, 0.5984, 0.3987, 0.2989]), 'bbox': [50.3750, 45.2707, 40.6427, 36.7410], 'keypoint': [59.9168, 53.4259, 47.2829, 42.0859]},
    'cheng2020': {'bpp': np.array([0.4891, 0.3064, 0.2173, 0.1460]), 'bbox': [50.0722, 47.3566, 44.5339, 40.4322], 'keypoint': [59.3911, 55.6319, 51.5906, 45.5846]},
    'color': 'black',
}


def calculate_bd_rate(x, anchor):
    for i in ['x264', 'x265', 'jpeg2000', 'cheng2020']:
        if i in x:
            x[i]['bd_rate'] = bj_delta(anchor[i]['bpp'], anchor[i]['keypoint'], x[i]['bpp'], x[i]['keypoint'], mode=1)
    x['bd_rate_hvs'] = (x['x264']['bd_rate'] + x['x265']['bd_rate'] + x['jpeg2000']['bd_rate'] + x['cheng2020']['bd_rate']) / 4.0


if __name__ == "__main__":
    print("keypoint detection: ")
    ap_list = []
    name_list = []
    with open("test.log") as f:
        lines = f.readlines()
        lines = [i.strip() for i in lines]
        for i in range(len(lines)):
            if "copypaste: Task: keypoints" in lines[i]:
                line = lines[i+2]
                ap = float(line.split("copypaste:")[1].split(",")[0])
                ap_list.append(ap)
            if "[Checkpointer] Loading from"  in lines[i]:
                name_list.append(lines[i].split("[Checkpointer] Loading from")[1])
    
    results = []
    for i in range(len(ap_list) // h // w):
        tmp = ap_list[:h*w]; ap_list = ap_list[h*w:]
        delta_bpp = 0.005
        current_result = {
            'x264': {'bpp': baseline['x264']['bpp']+delta_bpp, 'keypoint': np.array(tmp[:4])},
            'x265': {'bpp': baseline['x265']['bpp']+delta_bpp, 'keypoint': np.array(tmp[4:8])},
            'jpeg2000': {'bpp': baseline['jpeg2000']['bpp']+delta_bpp, 'keypoint': np.array(tmp[8:12])},
            'cheng2020': {'bpp': baseline['cheng2020']['bpp']+delta_bpp, 'keypoint': np.array(tmp[12:16])},
        }
        # print details 
        for method in ['x264', 'x265', 'jpeg2000', 'cheng2020']:
            print(f"{method}: {', '.join([str(i) for i in current_result[method]['keypoint']])}")
        # calculate and print BD-rate 
        calculate_bd_rate(current_result, baseline)
        print("name: {}, for baseline, bd_rate: {:.1f}%".format(name_list[i], current_result['bd_rate_hvs']))
        # calculate and print delta-AP
        delta_ap = 0.
        for method in ['x264', 'x265', 'jpeg2000', 'cheng2020']:
            delta_ap += np.mean(current_result[method]['keypoint'] - baseline[method]['keypoint'])
        delta_ap /= 4.
        print("delta ap: {:.1f}".format(delta_ap))
        print("")
    