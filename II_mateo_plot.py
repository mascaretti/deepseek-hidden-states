import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import List
import pandas as pd
import seaborn as sns
import typer
import scienceplots
import pickle

type="same"

np_file_list = [
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1024_AB.npy",
        "model_name": "DinoV2",
        "direction": "",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1025_AB.npy",
        "model_name": "DinoV2",
        "direction": "",
        "f_number": 1,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1026_AB.npy",
        "model_name": "DinoV2",
        "direction": "",
        "f_number": 2,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1027_AB.npy",
        "model_name": "DinoV2",
        "direction": "",
        "f_number": 3,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1028_AB.npy",
        "model_name": "DinoV2",
        "direction": "",
        "f_number": 4,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1024_AB.npy",
        "model_name": "iGPT",
        "direction": "",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1025_AB.npy",
        "model_name": "iGPT",
        "direction": "",
        "f_number": 1,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1026_AB.npy",
        "model_name": "iGPT",
        "direction": "",
        "f_number": 2,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1027_AB.npy",
        "model_name": "iGPT",
        "direction": "",
        "f_number": 3,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1028_AB.npy",
        "model_name": "iGPT",
        "direction": "",
        "f_number": 4,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1024_BA.npy",
        "model_name": "iGPT",
        "direction": "reciprocal",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1025_BA.npy",
        "model_name": "iGPT",
        "direction": "reciprocal",
        "f_number": 1,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1026_BA.npy",
        "model_name": "iGPT",
        "direction": "reciprocal",
        "f_number": 2,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1027_BA.npy",
        "model_name": "iGPT",
        "direction": "reciprocal",
        "f_number": 3,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/n_tokens_1028_BA.npy",
        "model_name": "iGPT",
        "direction": "reciprocal",
        "f_number": 4,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1024_BA.npy",
        "model_name": "DinoV2",
        "direction": "reciprocal",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1025_BA.npy",
        "model_name": "DinoV2",
        "direction": "reciprocal",
        "f_number": 1,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1026_BA.npy",
        "model_name": "DinoV2",
        "direction": "reciprocal",
        "f_number": 2,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1027_BA.npy",
        "model_name": "DinoV2",
        "direction": "reciprocal",
        "f_number": 3,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/n_tokens_1028_BA.npy",
        "model_name": "DinoV2",
        "direction": "reciprocal",
        "f_number": 4,
    },

]

np_file_list2 = [
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/clipped_II_AB_1.pkl",
        "model_name": "iGPT",
        "direction": "",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_igpt/clipped_II_BA_1.pkl",
        "model_name": "iGPT",
        "direction": "reciprocal",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/clipped_II_AB_1.pkl",
        "model_name": "DinoV2",
        "direction": "",
        "f_number": 0,
    },
        {
        "file": f"/home/mmahaut/projects/emecom/imagenet_{type}_class_dino2/clipped_II_BA_1.pkl",
        "model_name": "DinoV2",
        "direction": "reciprocal",
        "f_number": 0,
    },
]

np_file_list3 = [
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_same_class_dino2/cm_n_tokens_1024_AB.npy",
        "model_name": "dino->igpt",
        "direction": "",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/imagenet_same_class_dino2/cm_n_tokens_1024_BA.npy",
        "model_name": "igpt->dino",
        "direction": "",
        "f_number": 0,
    }
]

andrea_data=[
    {
        "file": "/home/mmahaut/projects/emecom/dino-jackknife/n_tokens_9_textimage.npy",
        "model_name": "DinoV2",
        "direction": "text->image",
        "f_number": 0,
    },
    {
        "file": "/home/mmahaut/projects/emecom/dino-jackknife/n_tokens_9_imagetext.npy",
        "model_name": "DinoV2",
        "direction": "image->text",
        "f_number": 0,
    },
    {
        "file": "/home/mmahaut/projects/emecom/igpt-jackknife/n_tokens_9_textimage.npy",
        "model_name": "iGPT",
        "direction": "text->image",
        "f_number": 0,
    },
    {
        "file": f"/home/mmahaut/projects/emecom/igpt-jackknife/n_tokens_9_imagetext.npy",
        "model_name": "iGPT",
        "direction": "image->text",
        "f_number": 0,
    }
]




def draw_plot2(output_file="./diagonal_plot.png"):
    # Global plot settings
    plt.style.use(['nature', 'science'])
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    # two subplots
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True)

    # DATA im-im
    df= pd.DataFrame()
    df["x"] = []
    df["II"] = []
    df["model_name"] = []
    df["direction"] = []

    for f in np_file_list:
        if ".npy" in f["file"]:
            II = np.load(f["file"])[1:, 0]
        elif ".pkl" in f["file"]:
            with open(f["file"], 'rb') as fi:
                II = pickle.load(fi)
                II = np.array([i[2] for i in II])
        rel_dep = np.linspace(0, 1, II.shape[0], endpoint=False)
        df = pd.concat([
            df,
            pd.DataFrame({
                "II": II,
                "x": rel_dep,
                "model_name": [f["model_name"]] * II.shape[0],
                "direction": [f["direction"]] * II.shape[0],
            })
        ], ignore_index=True)

    # Plot im-im
    average=df.groupby(["x", "model_name", "direction"]).mean().reset_index()
    variance=df.groupby(["x", "model_name", "direction"]).std().reset_index()


    for model_name in df["model_name"].unique():
        print(f"model_name: {model_name}",flush=True)
        for direction in df["direction"].unique():
            if direction == "reciprocal":
                continue
            print(f"model_name: {model_name}, direction: {direction}",flush=True)
            print(average.loc[(average["model_name"] == model_name) & (average["direction"] == direction), "x"])
            axs[1].plot(
                average.loc[(average["model_name"] == model_name) & (average["direction"] == direction), "x"],
                average.loc[(average["model_name"] == model_name) & (average["direction"] == direction), "II"],
                '-o',
                markersize=3,
                markerfacecolor='none' if direction == "reciprocal" else '#0A5EB0' if model_name == "DinoV2" else '#399918',
                color='#0A5EB0' if model_name == "DinoV2" else '#399918',
                markeredgecolor='#0A5EB0' if model_name == "DinoV2" else '#399918',
                markeredgewidth=0.8 if direction == "" else 0.3,
                label=f"{model_name} {'reciprocal' if direction == 'reciprocal' else ''}".strip(),
            )
            axs[1].fill_between(
                average.loc[(average["model_name"] == model_name) & (average["direction"] == direction), "x"],
                average.loc[(average["model_name"] == model_name) & (average["direction"] == direction), "II"] - variance.loc[(variance["model_name"] == model_name) & (variance["direction"] == direction), "II"],
                average.loc[(average["model_name"] == model_name) & (average["direction"] == direction), "II"] + variance.loc[(variance["model_name"] == model_name) & (variance["direction"] == direction), "II"],
                alpha=0.2,
                color='#0A5EB0' if model_name == "DinoV2" else '#399918',
                linewidth=0,
                label=None,
            )

    # Plot im-tok
    df = pd.DataFrame()
    df["x"] = []
    df["av_II"] = []
    df["std_II"] = []
    df["model_name"] = []
    df["direction"] = []
    for f in andrea_data:
        raw = np.load(f["file"])
        diags = raw[:,52,:]
        # diags = np.array([raw[:, round((i*61)/48), i] for i in range(raw.shape[2])])
        rel_dep = np.linspace(0, 1, diags.shape[-1], endpoint=False)
        df = pd.concat([
            df,
            pd.DataFrame({
                "av_II": np.mean(diags, axis=0),
                "std_II": np.std(diags, axis=0),
                "x": rel_dep,
                "model_name": [f["model_name"]] *  diags.shape[-1],
                "direction": [f["direction"]] *  diags.shape[-1],
            })
        ], ignore_index=True)
    print(df.head())
    print(df["av_II"].min(), df["av_II"].shape)
    for model_name in df["model_name"].unique():
        for direction in df["direction"].unique():
            axs[0].plot(
                df.loc[(df["model_name"] == model_name) & (df["direction"] == direction), "x"],
                df.loc[(df["model_name"] == model_name) & (df["direction"] == direction), "av_II"],
                '--o' if direction == "image->text" else '-o',
                linewidth= 0.3 if direction == "image->text" else 0.8,
                markersize=3,
                markerfacecolor='none' if direction == "image->text" else '#0A5EB0' if model_name == "DinoV2" else '#399918',
                color='#0A5EB0' if model_name == "DinoV2" else '#399918',
                markeredgecolor='#0A5EB0' if model_name == "DinoV2" else '#399918',
                markeredgewidth=0.3 if direction == "image->text" else 0.8,
                label=model_name if direction == 'text->image' else '',
            )
            axs[0].fill_between(
                df.loc[(df["model_name"] == model_name) & (df["direction"] == direction), "x"],
                df.loc[(df["model_name"] == model_name) & (df["direction"] == direction), "av_II"] - df.loc[(df["model_name"] == model_name) & (df["direction"] == direction), "std_II"],
                df.loc[(df["model_name"] == model_name) & (df["direction"] == direction), "av_II"] + df.loc[(df["model_name"] == model_name) & (df["direction"] == direction), "std_II"],
                alpha=0.2,
                color='#0A5EB0' if model_name == "DinoV2" else '#399918',
                linewidth=0,
                label=None,
            )
    axs[0].set_ylim(0.0, 1.0)
    axs[1].set_ylim(0.0, 1.0)
    axs[0].set_xlabel("Relative depth", fontsize=11)
    axs[1].set_xlabel("Relative depth", fontsize=11)
    axs[0].set_ylabel("Information imbalance", fontsize=11)
    axs[1].set_ylabel("", fontsize=11)
    axs[0].tick_params(axis='both', which='major', labelsize=11)
    axs[1].tick_params(axis='both', which='major', labelsize=11)
    axs[0].legend(fontsize=6.5, loc=3)
    axs[1].legend(fontsize=6.5, loc=3)

    # plt.xlabel("Relative depth", fontsize=11)
    # plt.ylabel("Information imbalance", fontsize=11)
    # plt.xticks(fontsize=11)
    # plt.yticks(fontsize=11)
    # plt.legend(fontsize=6.5, loc=3)
    # plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to {output_file}")
    plt.show()

# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt
# import os



def sparse_with_last(arr, every):
    arr = np.asarray(arr)
    sparse = arr[::every]
    
    if arr[-1] != sparse[-1]:
        sparse = np.append(sparse, arr[-1])
    
    return sparse

def draw_plot():
    # Load the .npy file
    plt.style.use(['nature', 'science'])

    plt.figure(figsize=(3.5, 2.))
    deepseek_english_spanish = np.load('/Users/riccardorende/Desktop/II_LLM/figure1/panel_b/results/n_tokens_30_AB_english_spanish.npy')
    deepseek_spanish_english = np.load('/Users/riccardorende/Desktop/II_LLM/figure1/panel_b/results/n_tokens_30_BA_english_spanish.npy')

    llama_english_spanish = np.loadtxt('/Users/riccardorende/Desktop/II_LLM/figure1/panel_a/llama_english_spanish_20.txt', usecols=1)
    llama_spanish_english = np.loadtxt('/Users/riccardorende/Desktop/II_LLM/figure1/panel_a/llama_spanish_english_20.txt', usecols=1)

    relative_depth_deepseek = np.arange(deepseek_english_spanish.size) / deepseek_english_spanish.size
    relative_depth_llama = np.arange(llama_english_spanish.size) / llama_english_spanish.size

    every = 4
    deepseek_english_spanish = sparse_with_last(deepseek_english_spanish, every)
    deepseek_spanish_english = sparse_with_last(deepseek_spanish_english, every)

    llama_english_spanish = sparse_with_last(llama_english_spanish, every)
    llama_spanish_english = sparse_with_last(llama_spanish_english, every)

    relative_depth_deepseek = sparse_with_last(relative_depth_deepseek, every)
    relative_depth_llama = sparse_with_last(relative_depth_llama, every)

    plt.plot(relative_depth_deepseek, deepseek_spanish_english, '-o', color="#0A5EB0", markersize=7, markerfacecolor='none', markeredgecolor='#0A5EB0', markeredgewidth=0.8)
    plt.plot(relative_depth_llama,llama_spanish_english, '-o', color="#399918", markersize=7, markerfacecolor='none', markeredgecolor='#399918', markeredgewidth=0.8)

    plt.plot(relative_depth_deepseek, deepseek_english_spanish, '-o', color="#0A5EB0", markersize=7, markeredgecolor='black', markeredgewidth=0.3, label="DeepSeek")
    plt.plot(relative_depth_llama, llama_english_spanish, '-o', color="#399918", markersize=7, markeredgecolor='black', markeredgewidth=0.3, label="Llama-3.1")

    plt.ylim(0.0, 1.0)

    plt.xlabel("Relative depth", fontsize=11)
    plt.ylabel("Information imbalance", fontsize=11)
    plt.title("English-spanish in different networks", fontsize=11)

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    plt.legend(fontsize=6.5, loc=3)
    plt.savefig("f1_a.pdf", dpi=300)

def draw_mins():
    plt.style.use(['nature', 'science'])
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering


    path="/home/mmahaut/projects/emecom/imagenet_same_class_dino2"
    files=Path(path).glob("n_tokens*.npy")
    path="/home/mmahaut/projects/emecom/imagenet_same_class_igpt"
    igpt_files=Path(path).glob("*n_tokens*.npy")
    files = sorted(files, key=lambda x: int(re.search(r"n_tokens_(\d+)_", str(x)).group(1)))[:62]
    igpt_files=sorted(igpt_files, key=lambda x: int(re.search(r"n_tokens_(\d+)_", str(x)).group(1)))[:147]
    AB_files = [file for file in files if "AB" in str(file)]
    AB_mins = [np.load(file).min() for file in AB_files]
    xs = [int(re.search(r"n_tokens_(\d+)_", str(file)).group(1)) for file in AB_files]
    igpt_AB_files = [file for file in igpt_files if "AB" in str(file)]
    # AB_mins = [AB_mins[i] for i in range(len(AB_mins)) if AB_mins[i] > 0.5]
    igpt_AB_mins = [np.load(file).min() for file in igpt_AB_files]
    igpt_xs = [int(re.search(r"n_tokens_(\d+)_", str(file)).group(1)) for file in igpt_AB_files]
    igpt_xs = [igpt_xs[i] for i in range(len(igpt_xs)) if igpt_AB_mins[i] > 0.5]
    igpt_AB_mins = [igpt_AB_mins[i] for i in range(len(igpt_AB_mins)) if igpt_AB_mins[i] > 0.5]
    # BA_mins = [np.load(file).min() for file in BA_files]
    # print(mins)
    plt.plot(xs, AB_mins, '-o', color="#0A5EB0", markersize=7, markeredgecolor='black', markeredgewidth=0.3, label="DinoV2")
    plt.plot(igpt_xs, igpt_AB_mins, '-o', color="#399918", markersize=7, markeredgecolor='black', markeredgewidth=0.3, label="iGPT")
    plt.xlabel("Number of tokens", fontsize=11)
    plt.ylabel("Min II", fontsize=11)
    plt.legend(fontsize=6.5, loc=3)
    save_path = "/home/mmahaut/projects/emecom/imagenet_same_class_dino2/mins.png"
    # save npy xs and mins
    np.save("/home/mmahaut/projects/emecom/imagenet_same_class_dino2/mins.npy", AB_mins)
    np.save("/home/mmahaut/projects/emecom/imagenet_same_class_dino2/xs.npy", xs)
    # save igpt npy xs and mins
    np.save("/home/mmahaut/projects/emecom/imagenet_same_class_igpt/mins.npy", igpt_AB_mins)
    np.save("/home/mmahaut/projects/emecom/imagenet_same_class_igpt/xs.npy", igpt_xs)
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

def draw_toks():
    plt.style.use(['nature', 'science'])
    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    path="/home/mmahaut/projects/emecom/imagenet_same_class_dino2"
    path2="/home/mmahaut/projects/emecom/imagenet_same_class_igpt"
    files=Path(path).glob("tt*.npy")
    files2=Path(path2).glob("tt*.npy")
    # print([(file,np.load(file).shape) for file in files])
    #(1,261)
    tcolors=["#0A5EB0", "#399918", "#FFA500"]
    colors={}
    ys={}
    xs2={}

    # two subplots
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 2.5), sharey=True)


    for _, layer_file in enumerate(sorted(files)):
        if "AB" in str(layer_file):
            continue
        tok_to_tok_II= np.load(layer_file)[0]
        xs = list(range(14))
        tok_xs=[j%14 for j in range(14)]
        tok_ys=[j//14 for j in range(14)]
        tok_dist = [np.sqrt((x - 0)**2 + (y - 0)**2) for x, y in zip(tok_xs, tok_ys)]
        # Group II values by token distance
        distance_dict = {}
        for ii, dist in zip(tok_to_tok_II, tok_dist):
            if dist not in distance_dict:
                distance_dict[dist] = []
            distance_dict[dist].append(ii)
        # Sort distances and compute mean II for each distance
        sorted_distances = sorted(distance_dict.keys())
        mean_ii_by_dist = [np.mean(distance_dict[d]) for d in sorted_distances]
        tok_to_tok_II = mean_ii_by_dist[1:]
        xs = sorted_distances[1:]
        layer = int(re.search(r"tt_layer_(\d+)_", str(layer_file)).group(1))
        if layer not in colors.keys():
            colors[layer] = tcolors[-1]
            tcolors.pop()
        # axs[0].plot(xs, tok_to_tok_II, '-o', color=colors[layer], markersize=3, markeredgecolor='black', markeredgewidth=0.3, label=layer)
        if layer not in ys.keys():
            ys[layer] = []
            xs2[layer] = []
        ys[layer].append(tok_to_tok_II)
        xs2[layer].append(xs)
    for layer in ys.keys():
        tok_to_tok_II = np.mean(ys[layer], axis=0)
        xs = np.mean(xs2[layer], axis=0)
        # tok_to_tok_II = [tok_to_tok_II[j] for j in range(len(tok_to_tok_II)) if j % 10 == 0]
        # xs = [xs[j] for j in range(len(xs)) if j % 10 == 0]
        axs[0].plot(xs, tok_to_tok_II, '-o', color=colors[layer], markersize=3, markeredgecolor='black', markeredgewidth=0.3, label=layer)
        # fill between std
        std = np.std(ys[layer], axis=0)
        # std = [std[j] for j in range(len(std)) if j % 10 == 0]
        # print(np.array(std).shape, np.array(xs).shape, np.array(tok_to_tok_II).shape)
        std = np.array(std)
        tok_to_tok_II = np.array(tok_to_tok_II)
        axs[0].fill_between(xs, tok_to_tok_II-std, tok_to_tok_II+std, alpha=0.2, color=colors[layer], linewidth=0)
    
    tcolors=["#0A5EB0", "#399918", "#FFA500"]
    colors={}
    ys={}
    xs2={}
    for _, layer_file in enumerate(sorted(files2)):
        if "AB" in str(layer_file):
            continue
        tok_to_tok_II= np.load(layer_file)[0]
        xs = list(range(32))
        # tok_to_tok_II = [tok_to_tok_II[j] for j in range(len(tok_to_tok_II)) if j % 10 == 0]
        # xs = [xs[j] for j in range(len(xs)) if j % 10 == 0]
        tok_xs=[j%32 for j in range(32)]
        tok_ys=[j//32 for j in range(32)]
        tok_dist = [np.sqrt((x - 0)**2 + (y - 0)**2) for x, y in zip(tok_xs, tok_ys)]
        # Group II values by token distance
        distance_dict = {}
        for ii, dist in zip(tok_to_tok_II, tok_dist):
            if dist not in distance_dict:
                distance_dict[dist] = []
            distance_dict[dist].append(ii)
        # Sort distances and compute mean II for each distance
        sorted_distances = sorted(distance_dict.keys())
        mean_ii_by_dist = [np.mean(distance_dict[d]) for d in sorted_distances]
        tok_to_tok_II = mean_ii_by_dist[1:]
        xs = sorted_distances[1:]

        layer = int(re.search(r"tt_layer_(\d+)_", str(layer_file)).group(1))
        if layer not in colors.keys():
            colors[layer] = tcolors[-1]
            tcolors.pop()

        if layer not in ys.keys():
            ys[layer] = []
            xs2[layer] = []
        ys[layer].append(tok_to_tok_II)
        xs2[layer].append(xs)

    for layer in ys.keys():
        tok_to_tok_II = np.mean(ys[layer], axis=0)
        xs = np.mean(xs2[layer], axis=0)
        # tok_to_tok_II = [tok_to_tok_II[j] for j in range(len(tok_to_tok_II)) if j % 10 == 0]
        # xs = [xs[j] for j in range(len(xs)) if j % 10 == 0]
        axs[1].plot(xs, tok_to_tok_II, '-o', color=colors[layer], markersize=3, markeredgecolor='black', markeredgewidth=0.3, label=layer)
        # fill between std
        std = np.std(ys[layer], axis=0)
        # std = [std[j] for j in range(len(std)) if j % 10 == 0]
        # print(np.array(std).shape, np.array(xs).shape, np.array(tok_to_tok_II).shape)
        std = np.array(std)
        tok_to_tok_II = np.array(tok_to_tok_II)
        axs[1].fill_between(xs, tok_to_tok_II-std, tok_to_tok_II+std, alpha=0.2, color=colors[layer], linewidth=0)

    axs[0].set_ylim(0.0, 1.19)
    axs[1].set_ylim(0.0, 1.19)
    axs[0].set_xlabel(r"$\tau$", fontsize=11)
    axs[1].set_xlabel(r"$\tau$", fontsize=11)
    axs[0].set_ylabel("Token-Token II", fontsize=11)
    axs[1].set_ylabel("Token-Token II", fontsize=11)
    axs[0].tick_params(axis='both', which='major', labelsize=11)
    axs[1].tick_params(axis='both', which='major', labelsize=11)
    axs[0].legend(fontsize=6.5, loc=3)
    axs[1].legend(fontsize=6.5, loc=3)

    plt.savefig(Path(path)/f"tok_to_tok_II_layer.png", dpi=300)
    plt.close()
    print(f"Saved plot to {Path(path)/f'tok_to_tok_II_layer.png'}")



if __name__ == "__main__":
    typer.run(draw_toks)