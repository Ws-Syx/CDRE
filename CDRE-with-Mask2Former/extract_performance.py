import matplotlib.pyplot as plt
import numpy as np
import os, sys, json
import matplotlib
# matplotlib.rcParams['font.family'] = 'Times New Roman'

codec_list = ['dcvc_dc', 'dcvc', 'x265', 'x264']


# initialize some import data
if True:
    baseline = {
        'dcvc_dc':{'bpp': [0.203, 0.139, 0.094, 0.065 ], 'ap': [51.026, 51.306, 50.181, 47.969]},
        'dcvc':{'bpp': [0.188, 0.131, 0.090, 0.062 ], 'ap': [50.650, 50.157, 49.738, 47.453]},
        'x265':{'bpp': [0.151, 0.102, 0.069, 0.047], 'ap': [50.997, 49.889, 49.772, 48.196]},
        'x264':{'bpp': [0.168, 0.122, 0.089, 0.065], 'ap': [50.833, 50.231, 49.483, 47.642]},
        'color': 'black',
    }

    # train-5-3
    full_finetune = {
        'dcvc_dc':{'bpp': [0.203, 0.139, 0.094, 0.065 ], 'ap': [52.231, 52.362, 51.410, 50.747]},
        'dcvc':{'bpp': [0.188, 0.131, 0.090, 0.062 ], 'ap': [51.801, 51.196, 50.987, 49.840]},
        'x265':{'bpp': [0.151, 0.102, 0.069, 0.047], 'ap': [52.055, 51.088, 51.031, 49.556]},
        'x264':{'bpp': [0.168, 0.122, 0.089, 0.065], 'ap': [52.085, 51.941, 49.627, 49.395]},
        'color': 'blue',
    }

    # train-9
    head_finetune = {
        'dcvc_dc':{'bpp': [0.203, 0.139, 0.094, 0.065 ], 'ap': [51.515, 51.990, 51.212, 50.467]},
        'dcvc':{'bpp': [0.188, 0.131, 0.090, 0.062 ], 'ap': [51.740, 51.305, 50.660, 48.443]},
        'x265':{'bpp': [0.151, 0.102, 0.069, 0.047], 'ap': [52.134, 51.069, 51.141, 49.354]},
        'x264':{'bpp': [0.168, 0.122, 0.089, 0.065], 'ap': [51.915, 51.542, 50.526, 49.014]},
        'color': 'green',
    }

    # PromptIR
    promptIR = {
        'dcvc_dc':{'bpp': [0.203, 0.139, 0.094, 0.065 ], 'ap': [50.682, 50.718, 49.653, 47.175]},
        'dcvc':{'bpp': [0.188, 0.131, 0.090, 0.062 ], 'ap': [50.502, 49.891, 49.369, 47.100]},
        'x265':{'bpp': [0.151, 0.102, 0.069, 0.047], 'ap': [51.087, 50.591, 49.085, 47.994]},
        'x264':{'bpp': [0.168, 0.122, 0.089, 0.065], 'ap': [50.620, 50.292, 49.354, 47.873]},
        'color': 'orange',
    }

    # PromptCIR
    promptCIR = {
        'dcvc_dc':{'bpp': [0.203, 0.139, 0.094, 0.065 ], 'ap': [51.351, 51.294, 50.182, 47.719]},
        'dcvc':{'bpp': [0.188, 0.131, 0.090, 0.062 ], 'ap': [50.874, 50.976, 49.819, 47.835]},
        'color': 'purple',
    }


def draw_performance(data, output_path):
    # extra bpp for distortion stream
    delta_bpp = 0.001573
    for codec_name in codec_list:
        data['ours'][codec_name]['bpp'] = [i + delta_bpp for i in data['ours'][codec_name]['bpp']]

    plt.figure(figsize=(8, 6.5), dpi=150)
    for i, codec_name in enumerate(codec_list):
        plt.subplot(2, 2, i + 1)
        for method in data:
            plt.plot(data[method][codec_name]['bpp'], data[method][codec_name]['ap'], linestyle='-', marker='o', color=data[method]['color'], label=method)
        plt.legend(loc='lower right')
        plt.grid(linestyle='-', alpha=0.25)
        plt.title(codec_name)
        plt.savefig(output_path)


if __name__ == "__main__":

    os.system("rm -rf ./performance")
    os.system("mkdir ./performance")

    ap_list = []
    name_list = []
    with open("test.log") as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if "|:------:|:------:|:------:|:------:|:------:|:------:|:------:|:------:|" in lines[i]:
                line = lines[i+1]
                ap = float(line.split(" ")[1])
                ap_list.append(ap)
            if "./ckpt/" in lines[i] and ".pth" in lines[i] and "Loading from" in lines[i]:
                name = lines[i].split("./ckpt/")[1].split(".pth")[0]
                if name not in name_list:
                    name_list.append(name)
    # name_list = list(set(name_list))  
    print(name_list)
    
    results = []
    for i in range(len(ap_list) // 16):
        tmp = ap_list[:16]; ap_list = ap_list[16:]
        current_result = {
            'dcvc_dc': {'bpp': baseline['dcvc_dc']['bpp'], 'ap': tmp[:4]},
            'dcvc': {'bpp': baseline['dcvc']['bpp'], 'ap':  tmp[4:8]},
            'x265': {'bpp': baseline['x265']['bpp'], 'ap': tmp[8:12]},
            'x264': {'bpp': baseline['x264']['bpp'], 'ap': tmp[12:16]},
            'color': 'red'
        }
        print("name: ", name_list[i]); print(current_result)

        # delta ap for baseline
        delta_ap = np.mean([(np.array(current_result[codec_name]['ap']) - np.array(baseline[codec_name]['ap'])) / np.array(baseline[codec_name]['ap']) for codec_name in codec_list])
        delta_ap = round(delta_ap * 100, 3)
        print(f"name: {name_list[i]}, delta AP for baseline: {delta_ap}%")

        # delta ap for full-finetune
        delta_ap = np.mean([(np.array(current_result[codec_name]['ap']) - np.array(full_finetune[codec_name]['ap'])) / np.array(full_finetune[codec_name]['ap']) for codec_name in codec_list])
        delta_ap = round(delta_ap * 100, 3)
        print(f"name: {name_list[i]}, delta AP for finetune: {delta_ap}%")
        print("")
        
        draw_performance({'baseline': baseline,
                          'head_finetune': head_finetune,
                          'full_finetune': full_finetune,
                          'promptIR': promptIR,
                          'ours': current_result},
                          output_path=f"./performance/{name_list[i].replace('/', ' ')}.png")