import os, sys, tqdm
import numpy as np

h = w = 4

original = {
    'dcvc_dc': np.array([34.8873, 34.2441, 32.5336, 30.6322]),
    'dcvc': np.array([34.2413, 32.8917, 31.2756, 28.7281]),
    'x265': np.array([35.8086, 35.1047, 34.2413, 32.5263]),
    'x264': np.array([36.0740, 35.17769, 34.1900, 31.9097]),
}

if __name__ == "__main__":
    print("bbox: ")
    ap_list = []
    with open("test.log") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if ": copypaste: AP,AP50,AP75,APs,APm,APl" in lines[i]:
                line = lines[i+1]
                ap = float(line.split("copypaste:")[1].split(",")[0])
                ap_list.append(ap)
    
    results = []
    for i in range(len(ap_list) // h // w):
        tmp = ap_list[:h*w]; ap_list = ap_list[h*w:]
        current_result = {
            'x264': np.array(tmp[:4]),
            'x265': np.array(tmp[4:8]),
            'jpeg2000': np.array(tmp[8:12]),
            'cheng2020': np.array(tmp[12:]),
        }
        
        for method in current_result:
            print(f"{method}: {', '.join([str(i) for i in current_result[method]])}")


    print("keypoint detection: ")
    ap_list = []
    with open("test.log") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if ": copypaste: AP,AP50,AP75,APm,APl" in lines[i]:
                line = lines[i+1]
                ap = float(line.split("copypaste:")[1].split(",")[0])
                ap_list.append(ap)
    
    results = []
    for i in range(len(ap_list) // h // w):
        tmp = ap_list[:h*w]; ap_list = ap_list[h*w:]
        current_result = {
            'x264': np.array(tmp[:4]),
            'x265': np.array(tmp[4:8]),
            'jpeg2000': np.array(tmp[8:12]),
            'cheng2020': np.array(tmp[12:]),
        }
        
        for method in current_result:
            print(f"{method}: {', '.join([str(i) for i in current_result[method]])}")

    