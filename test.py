import os
import glob
import imageio
import pandas as pd
from proc_img import *
import matplotlib.pyplot as plt


def main():
    for i, path in enumerate(glob.glob("exemplars/reflacx/gaze_data/*-*-*-*-*_P*R*/gaze.csv")):
        df = pd.read_csv(path)
        df = df.loc[:, "window_width":"window_level"].iloc[::(len(df.index) // 10)]

        path = os.path.splitext(path)[0]
        path = path[(path.find("\\") + 1):(path.rfind("\\") - 12)]

        proc = ProcImg(["exemplars/images_dcm/" + path + ".dcm"] * len(df.index))

        os.makedirs("data_playground/" + path, exist_ok=True)

        for j, (k, row) in enumerate(df.iterrows()):
            width, level = row

            proc.imgs[j] = window_img(proc.imgs[j], level=level, width=width)

            plt.imshow(proc.imgs[j], cmap="gray")
            plt.savefig("data_playground/" + path + "/" + str(j) + ".png")

            proc.imgs[j] = imageio.v2.imread("data_playground/" + path + "/" + str(j) + ".png")

        imageio.mimsave("data_playground/" + path + "/xray.gif", proc.imgs, fps=2)


if __name__ == "__main__":
    main()
