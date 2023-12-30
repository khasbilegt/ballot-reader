import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

logging.basicConfig(
    filename="./election.log",
    filemode="w",
    format="[%(asctime)s] %(levelname)s - %(message)s",
)


def pre_process_image(img):
    # kernel = np.ones((5, 5), np.uint8)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # blur = cv.GaussianBlur(gray, (9, 9), 0)
    # blur = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)
    _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    return threshold


def find_contours(img, area_threshold=(1200, 2000), arc_threshold=(100, 200)):
    height, width = img.shape
    image = img.copy()

    filtered_contours = []
    for area in [
        image[0:200, 0:200],
        image[0:200, width - 200 : width],
        image[height - 200 : height, 0:200],
    ]:
        contours, _ = cv.findContours(area, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        convexed_contours = [cv.convexHull(c) for c in contours]

        rectangles = []
        for c in convexed_contours:
            polygon = cv.approxPolyDP(c, 0.1 * cv.arcLength(c, True), True)
            if len(polygon) == 4:
                rectangles.append(polygon)

        sorted_rectangles = sorted(
            rectangles, key=lambda x: cv.contourArea(x), reverse=True
        )
        filtered_contours.append(sorted_rectangles[0])

    right_contour = np.array(
        [
            [[point[0][0] + (width - 200), point[0][1]]]
            for point in filtered_contours[1].tolist()
        ],
        np.int32,
    )
    bottom_contour = np.array(
        [
            [[point[0][0], point[0][1] + (height - 200)]]
            for point in filtered_contours[2].tolist()
        ],
        np.int32,
    )
    return [filtered_contours[0], right_contour, bottom_contour]


def calculate_edge_points(contours):
    points = np.concatenate(contours)
    hull = cv.convexHull(points)
    epsilon = 0.1 * cv.arcLength(hull, True)
    approximated_hull = cv.approxPolyDP(hull, epsilon, True)

    top_left, top_right, bottom_left = sorted(
        [(point[0][0], point[0][1]) for point in approximated_hull.tolist()],
        key=lambda p: sum(p),
    )
    bottom_right = (top_right[0], bottom_left[1])
    return top_left, top_right, bottom_left, bottom_right


def crop_image(img):
    image = pre_process_image(img)
    contours = find_contours(image)
    top_left, top_right, bottom_left, bottom_right = calculate_edge_points(contours)

    width = max(top_right[0] - top_left[0], bottom_right[0] - bottom_left[0])
    height = max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])

    source = np.float32([top_left, top_right, bottom_left, bottom_right])
    dest = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv.getPerspectiveTransform(source, dest)
    cropped = cv.warpPerspective(image, matrix, (width, height))

    return cropped


def is_filled(img):
    whites = np.sum(img == 255)
    blacks = np.sum(img == 0)
    return whites > (whites + blacks) * 0.3


def get_votes(img, candidates, x_weight=0, y_weight=0):
    vote_count = 0
    votes = []
    cropped = np.array(img, copy=True)

    for group, name, x, y in candidates:
        x_start = 66 + x_weight
        x_step = 11
        y_start = 54 + y_weight
        y_step = 12

        left = x_start + 2 * x_step * (x - 1)
        right = left + x_step * 2
        top = y_start + 2 * y_step * (y - 1)
        bottom = top + y_step

        color = (255, 0, 0)
        y_half = int(y_step / 2)
        origin = [left + x_step, top + y_half]

        # cv.rectangle(cropped, [left, top], [right, bottom], color, 1)
        cv.ellipse(cropped, origin, [x_step, y_half], 0, 0, 360, color, 1)

        voted = is_filled(img[top:bottom, left + 5 : right - 5])
        if voted:
            vote_count += 1
        votes.append((group, name, voted))

    return votes, cropped, vote_count


def detect_votes(path, metadata, write_dir=Path("./process")):
    try:
        img = cv.imread(str(path))
        cropped_image = crop_image(img)
        votes, marked_image, vote_count = get_votes(
            cropped_image, metadata["candidates"]
        )
        converted_image = cv.cvtColor(marked_image, cv.COLOR_GRAY2BGR)

        if not (vote_count == metadata["quota"]):
            logging.error(
                f"Буруу тоолсон: {vote_count}/{metadata["quota"]} - {path}",
            )
            # print("\tСаналууд: ")
            # for group, name, vote in votes:
            #     print(f"\t\t{"🟢" if vote else "🔴"} {name} [{group}]")
            # print("\n")

            if not write_dir.exists():
                write_dir.mkdir()

            height, width, _ = converted_image.shape

            cv.putText(
                converted_image,
                f"Counted: {vote_count}",
                (int(width / 2) - 100, height - 250),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv.LINE_AA,
            )
            cv.putText(
                converted_image,
                f"Quota: {metadata["quota"]}",
                (int(width / 2) - 100, height - 200),
                cv.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
            # cv.imwrite(str(write_dir / f"{path.stem}_cropped.jpeg"), cropped_image)
            cv.imwrite(str(write_dir / f"{path.stem}.jpeg"), converted_image)

        return votes
    except AssertionError:
        logging.error(f"Булан танилтууд алдаатай - {path}")


def get_metadata(path, row_offset=9, column_offset=11):
    if (
        path := Path((path if path.is_dir() else path.parent) / "metadata.yaml")
    ) and not path.exists():
        raise FileNotFoundError(
            "Нэр дэвшигчдийн мэдээлэл одсонгүй. Боловсруулалт хийх файлын хамт metadata.yaml гэсэн нэртэйгээр оруулна уу."
        )

    with path.open(mode="rb") as fp:
        config = yaml.load(fp, Loader=yaml.Loader)

    candidates = []
    for group_name, data in config["groups"].items():
        row = data["row"] + row_offset
        column = column_offset * (data["column"] - 1) + 1
        for index, candidate in enumerate(data["candidates"]):
            candidates.append((group_name, candidate, column, row + index))
    return {"quota": config["quota"], "candidates": candidates}


def process_path(path, metadata_path):
    if path.is_dir():
        print("📁 Хавтас: ", path)
        filepaths = [
            Path(root, f)
            for root, _, files in path.walk(on_error=print)
            for f in files
            if Path(root, f).suffix in [".jpeg", ".jpg"]
        ]
        total = len(filepaths)
        metadata = get_metadata(path)
        for index, path in enumerate(filepaths, start=1):
            if votes := detect_votes(path, metadata):
                print(
                    f"{index}/{total} Тоолсон: {len([v for v in votes if v[-1]])} Хүчинтэй: {metadata["quota"]}"
                )
    else:
        print("📄 Файл: ", path)
        metadata = get_metadata(metadata_path)
        votes = detect_votes(path, metadata)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Боловсруулалт хийх зураг эсвэл хавтсын замыг оруулна уу!")

    if len(sys.argv) >= 2 and (path := Path(sys.argv[1])) and not path.exists():
        raise FileNotFoundError("Заасан зам олдсонгүй!")

    if len(sys.argv) == 3:
        metadata = sys.argv[2]
        process_path(path, Path(metadata))
    # process_path(path)
