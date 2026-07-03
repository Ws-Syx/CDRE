import os, sys, tqdm
import numpy as np

h = w = 4

original = {
    'dcvc_dc': np.array([51.026, 51.306, 50.181, 47.969]),
    'dcvc': np.array([50.650, 50.157, 49.738, 47.453]),
    'x265': np.array([50.997, 49.889, 49.772, 48.196]),
    'x264': np.array([50.833, 50.231, 49.483, 47.642]),
}

# full_finetune = {
#     'dcvc_dc': [52.231, 52.362, 51.41 , 50.747],
#     'dcvc': [51.801, 51.196, 50.987, 49.84 ],
#     'x265': [52.055, 51.088, 51.031, 49.556],
#     'x264': [52.085, 51.941, 49.627, 49.395],
# }

if __name__ == "__main__":
    ap_list = []
    with open("test.log") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|" in lines[i]:
                line = lines[i+1]
                ap = float(line.split(" ")[1])
                ap_list.append(ap)
    
    # assert len(ap_list) % (h * w) == 0, f"error: len(ap_list) is {len(ap_list)}"
    
    results = []
    for i in range(len(ap_list) // h // w):
        tmp = ap_list[:h*w]; ap_list = ap_list[h*w:]
        current_result = {
            'dcvc_dc': np.array(tmp[:4]),
            'dcvc': np.array(tmp[4:8]),
            'x265': np.array(tmp[8:12]),
            'x264': np.array(tmp[12:16]),
        }
        
        for method in current_result:
            print(f"{method}: {', '.join([str(i) for i in current_result[method]])}")
        delta_ap = np.mean([(current_result[method] - original[method]) / original[method] for method in current_result])
        delta_ap = round(delta_ap * 100, 3)
        print(f"i: {i}, delta AP: {delta_ap}%")
        print("")
