import json
from re import split
import shutil
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import io
from shapely.geometry import Polygon

Image.MAX_IMAGE_PIXELS = None


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        shutil.rmtree(path)
        os.makedirs(path)


def dice(a, b):
    return 2 * a.intersection(b).area / (a.area + b.area)


def recall(a, b):
    return a.intersection(b).area / b.area


def precision(a, b):
    return a.intersection(b).area / a.area


def find_diff(dice_thred=0.5, draw_preview=True, log_score=True):
    # A - new json
    with open(file_A_path) as data_file:
        data = json.load(data_file)

    average_area = sum(
        [Polygon(item["geometry"]["coordinates"][0]).area for item in data]
    ) / len(data)
    area_threshold = average_area / 50
    print("average area size: ", average_area)
    print("size threshold: ", area_threshold)

    coor_list_a = []

    for item in data:
        coor = item["geometry"]["coordinates"]
        poly = Polygon(coor[0])
        if poly.area > area_threshold:
            coor_list_a.extend(item["geometry"]["coordinates"])
        else:
            print("A ignore", poly.area)
    A_x_list = [[xy[0] for xy in coor] for coor in coor_list_a]
    A_y_list = [[xy[1] for xy in coor] for coor in coor_list_a]
    A_id_list = [i for i in range(len(coor_list_a))]

    # B - old json
    with open(file_B_path) as data_file:
        data = json.load(data_file)

    coor_list_b = []

    for item in data:
        coor = item["geometry"]["coordinates"]
        # for some json. Comment this line if needed
        # coor = [[[xy[1], xy[0]] for xy in coor[0]]]

        poly = Polygon(coor[0])
        if poly.area > area_threshold:
            coor_list_b.extend(coor)
        else:
            print("B ignore", poly.area)
    B_x_list = [[xy[0] for xy in coor] for coor in coor_list_b]
    B_y_list = [[xy[1] for xy in coor] for coor in coor_list_b]

    # find difference
    center_list_new = []
    for i in range(len(A_x_list)):
        mean_x = (sum(A_x_list[i]) - A_x_list[i][-1]) / (len(A_x_list[i]) - 1)
        mean_y = (sum(A_y_list[i]) - A_y_list[i][-1]) / (len(A_y_list[i]) - 1)
        center_list_new.append((mean_x, mean_y))

    center_list_old = []
    for i in range(len(B_x_list)):
        mean_x = (sum(B_x_list[i]) - B_x_list[i][-1]) / (len(B_x_list[i]) - 1)
        mean_y = (sum(B_y_list[i]) - B_y_list[i][-1]) / (len(B_y_list[i]) - 1)
        center_list_old.append((mean_x, mean_y))

    new_added_list = []
    new_added_f1_list = []
    new_same_list = []
    new_revised_list = []
    f1_list = []

    positon_threshold = 500
    dice_threshold = dice_thred

    ignore_count = 0
    for i in A_id_list:
        x, y = center_list_new[i]
        new_p = Polygon(coor_list_a[i])
        min_f1 = 0
        min_j = -1
        _recall, _precision = -1, -1
        for j in range(len(center_list_old)):
            _x, _y = center_list_old[j]
            old_p = Polygon(coor_list_b[j])
            if (x - _x) ** 2 + (y - _y) ** 2 <= positon_threshold ** 2:
                f1 = dice(new_p, old_p)
                if f1 > min_f1:
                    min_f1 = f1
                    min_j = j
                    _recall = recall(new_p, old_p)
                    _precision = precision(new_p, old_p)

        if min_f1 >= 0.999:
            _flag = f"Same\t{min_f1}"
            new_same_list.append(i)
        elif min_f1 >= dice_threshold:
            _flag = f"Revised\t{min_f1}"
            new_revised_list.append(i)
            f1_list.append((min_f1, _recall, _precision))
        else:
            _flag = f"Added\t{min_f1}"
            new_added_list.append(i)
            new_added_f1_list.append(min_f1)
            # print(min_f1)
        if _flag.startswith("Same") or _flag.startswith("Revised"):
            if min_j != -1:
                coor_list_b.pop(min_j)
                center_list_old.pop(min_j)
        # print(i, _flag)

    removed_count = len(center_list_old)
    print(f"A\tB\tsame\tmatch\tadded\tdeleted")
    print(
        f"{len(A_x_list)}\t{len(B_x_list)}\t{len(new_same_list)}\t{len(new_revised_list)}"
        f"\t{len(new_added_list)}\t{removed_count}"
    )
    print(f"[FP: {len(new_added_list)}/{len(A_x_list)}]")
    print(f"[FN: {removed_count}/{len(B_x_list)}]")
    # print(f"{len(new_same_list)} same")
    # print(f"{len(new_revised_list)} revised")
    # print(f"{len(new_added_list)} added")
    # print(f"{removed_count} deleted")

    # draw visualization
    if draw_preview:
        ref_image = io.imread(image_ref_path)
        background = np.zeros(shape=ref_image.shape, dtype=np.uint8)
        img = Image.fromarray(background, "L")
        img = img.convert("RGB")
        font_path = r"c:\windows\fonts\bahnschrift.ttf"
        font = ImageFont.truetype(font_path, size=48)
        title_font = ImageFont.truetype(font_path, size=72)
        ImageDraw.Draw(img).text(
            (100, 400),
            text=f"DICE Threshold = {dice_thred}",
            font=title_font,
            fill="white",
        )
        ImageDraw.Draw(img).text(
            (100, 480),
            text=f"PREDICTION [FP: {len(new_added_list)}/{len(A_x_list)}]",
            font=title_font,
            fill="yellow",
        )
        ImageDraw.Draw(img).text(
            (100, 560),
            text=f"GROUND TRUTH [FN: {removed_count}/{len(B_x_list)}]",
            font=title_font,
            fill="red",
        )

        for i in new_added_list:
            coor_tuple = [(xy[1], xy[0]) for xy in coor_list_a[i]]
            # print(coor_tuple)
            ImageDraw.Draw(img).line(coor_tuple, fill="yellow", width=6)
            # text
            f1 = new_added_f1_list[new_added_list.index(i)]
            if f1 > 0:
                text = "{:.3f}".format(f1)  # + f",{Polygon(coor_list_a[i]).area}"
                ImageDraw.Draw(img).text(
                    (center_list_new[i][1] - 40, center_list_new[i][0] + 60),
                    text,
                    font=font,
                )

        for coor_b in coor_list_b:
            coor_tuple = [(xy[1], xy[0]) for xy in coor_b]
            # print(coor_tuple)
            ImageDraw.Draw(img).line(coor_tuple, fill="red", width=6)
            # text = f",{Polygon(coor_b).area}"
            # ImageDraw.Draw(img).text(
            #     (coor_tuple[0][0], coor_tuple[0][1]),
            #     text,
            #     font=font,
            # )
        img = np.array(img).astype("uint8")
        output_path = image_ref_path.replace(
            ".png", f'_{str(dice_thred).replace(".","_")}.png'
        )
        io.imsave(output_path, img)
        print(f"Image saved to {output_path}")

    # write score
    if log_score:
        txt_path = file_A_path.replace("json", "txt")
        with open(txt_path, "w") as f:
            for item in f1_list:
                f.write(f"{item[0]},{item[1]},{item[2]}\n")


if __name__ == "__main__":
    file_A_path = (
        r"C:\Users\yiju\Desktop\Copy\Scripts\masks\1-tom-new-kidney\pred_00a67c839.json"
    )
    file_B_path = r"C:\Users\yiju\Desktop\Copy\Data\hubmap-kidney-segmentation\test\00a67c839.json"

    if len(sys.argv) >= 3:
        file_A_path = sys.argv[1]
        file_B_path = sys.argv[2]
    image_ref_path = file_A_path.replace("json", "png")

    A_name = file_A_path.split("\\")[-1].split(".")[0]
    B_name = file_B_path.split("\\")[-1].split(".")[0]
    print("A: ", A_name)
    print("B: ", B_name)

    for d in [0.5]:  # [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        find_diff(dice_thred=d, draw_preview=True, log_score=True)
