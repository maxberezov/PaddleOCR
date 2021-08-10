from paddleocr import PaddleOCR, draw_ocr
import sys, os
import json
from PIL import Image
from typing import List, Dict, Tuple
import re

ocr = PaddleOCR(lang="en")
image_path = sys.argv[1]
output_path = sys.argv[2]


def rotate_img(image_path: str, rt_degr: int):
    img = Image.open(image_path)
    return img.rotate(rt_degr, expand=False)


def get_rotations(image_path: str, rotation_required=False) -> List[str]:
    if rotation_required:
        paths = []
        for degree in range(0, 360, 45):
            rotated = rotate_img(image_path, degree)
            extension = image_path[image_path.find('.'):]
            path = os.path.join(image_path[:image_path.find('.')]) + '_{}{}'.format(str(degree), extension)
            rotated.save(path)
            paths.append(path)
        return paths
    else:
        return [image_path]


def get_predictions(paths: List[str], threshold=0.85, rotation_required=False) -> List[Dict]:
    output = []
    for p in paths:
        result = ocr.ocr(p)
        angle = 0
        if rotation_required:
            angle = int(p[p.rfind('_') + 1:p.rfind('.')])
        for line in result:
            proba = line[1][1]
            if proba > threshold:
                output.append({"extracted_text": line[1][0], "coordinates": line[0], "proba": proba, "angle": angle,
                               "attributes": {"IsRadius": False, "required_measurement": False, "lower_bound": None,
                                              "upper_bound": None}})
    return output


def process_radius_attribute(output: List[Dict]) -> None:
    for res in output:
        if re.search('R[0-9]+', res["extracted_text"]) is not None:
            res["attributes"]["IsRadius"] = True


def process_m_attribute(output: List[Dict]):
    for i in range(len(output)):
        for j in range(len(output)):
            if check_m_condition(output[i]["coordinates"], output[j]["coordinates"]):
                output[j]["attributes"]["required_measurement"] = output[i]["extracted_text"]


def process_bound_attribute(output: List[Dict]):
    for i in range(len(output)):
        for j in range(len(output)):
            for k in range(len(output)):
                if check_bound_attributes(output[i]["coordinates"], output[j]["coordinates"], output[k]["coordinates"]):
                    output[k]["attributes"]["lower_bound"] = output[j]["extracted_text"]
                    output[k]["attributes"]["upper_bound"] = output[i]["extracted_text"]


def save_to_json(output_path: str, output: List[Dict]) -> None:
    with open(output_path, 'w') as f:
        json.dump(str(output), f)


def main():
    paths = get_rotations(image_path)
    output = get_predictions(paths)
    process_radius_attribute(output)
    process_m_attribute(output)
    process_bound_attribute(output)
    save_to_json(output_path, output)


def get_horizontal_coordinates(a: List[List[float]]) -> Tuple[float, float]:
    x = [elem[0] for elem in a]
    x.sort()
    return (x[0] + x[1]) / 2, (x[2] + x[3]) / 2


def get_vertical_coordinates(a: List[List[float]]) -> Tuple[float, float]:
    y = [elem[1] for elem in a]
    y.sort()
    return (y[0] + y[1]) / 2, (y[2] + y[3]) / 2


def compute_square(x: Tuple[float, float], y: Tuple[float, float]) -> float:
    return (x[1] - x[0]) * (y[1] - y[0])


def check_vertical_correspondence(y1: Tuple[float, float], y2: Tuple[float, float], empirical_const=0.2) -> bool:
    diff_1 = abs(y1[0] - y2[0])
    diff_2 = abs(y1[1] - y2[1])
    size = y1[1] - y1[0]
    return diff_1 < empirical_const * size and diff_2 < empirical_const * size


def check_right_margin(x1: Tuple[float, float], x2: Tuple[float, float]) -> bool:
    diff = x2[0] - x1[1]
    size = x1[1] - x1[0]
    if 0 < diff < size:
        return True
    return False


def check_m_condition(a: List[List[float]], b: List[List[float]]) -> bool:
    x1 = get_horizontal_coordinates(a)
    y1 = get_vertical_coordinates(a)
    x2 = get_horizontal_coordinates(b)
    y2 = get_vertical_coordinates(b)
    return check_right_margin(x1, x2) and check_vertical_correspondence(y1, y2)


def check_area_condition(x1: Tuple[float, float], x2: Tuple[float, float], x3: Tuple[float, float],
                         y1: Tuple[float, float], y2: Tuple[float, float], y3: Tuple[float, float]) -> bool:
    return compute_square(x3, y3) > 1.3 * compute_square(x1, y1) and compute_square(x3, y3) > 1.3 * compute_square(x2,
                                                                                                                   y2)


def check_upper_bound(pivot: Tuple[float, float], upper: Tuple[float, float], empirical_const=0.9) -> bool:
    middle_point = pivot[0] + pivot[1]
    middle_point /= 2
    if pivot[1] > upper[0] > empirical_const * middle_point:
        return True
    return False


def check_lower_bound(pivot: Tuple[float, float], lower: Tuple[float, float], empirical_const=1.1) -> bool:
    middle_point = pivot[0] + pivot[1]
    middle_point /= 2
    if pivot[0] < lower[1] < empirical_const * middle_point:
        return True
    return False


def check_bound_attributes(a: List[List[float]], b: List[List[float]], c: List[List[float]]) -> bool:
    x1 = get_horizontal_coordinates(a)
    y1 = get_vertical_coordinates(a)
    x2 = get_horizontal_coordinates(b)
    y2 = get_vertical_coordinates(b)
    y3 = get_vertical_coordinates(c)
    x3 = get_horizontal_coordinates(c)

    return check_area_condition(x1, x2, x3, y1, y2, y3) and check_lower_bound(y3, y1) and check_upper_bound(y3,
                                                                                                            y2) and check_right_margin(
        x3, x1) and check_right_margin(x3, x2)


if __name__ == '__main__':
    main()
