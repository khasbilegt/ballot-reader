import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

logging.basicConfig(filename="./election.log")
logger = logging.getLogger()


class BallotException(Exception):
    quota: int
    counted: int

    def __init__(self, quota: int, counted: int, *args: object) -> None:
        self.quota = quota
        self.counted = counted

        return super().__init__(
            f"Боловсруулалтын үр дүн ({counted}), дугуйлсан нэр дэвшигчийн тоо ({quota}) таарсангүй.",
            *args,
        )


def pre_process_image(img):
    # kernel = np.ones((5, 5), np.uint8)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # blur = cv.GaussianBlur(gray, (9, 9), 0)
    # blur = cv.morphologyEx(blur, cv.MORPH_CLOSE, kernel)
    _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    return threshold


def find_contours(img, area_threshold=(1200, 2000), arc_threshold=(100, 200)):
    contours, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        arc = cv.arcLength(contour, True)

        if (
            area > min(area_threshold)
            and area < max(area_threshold)
            and arc > min(arc_threshold)
            and arc < max(arc_threshold)
        ):
            filtered_contours.append(contour)

    assert len(filtered_contours) == 3
    return filtered_contours


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

        voted = is_filled(img[top:bottom, left:right])
        votes.append((group, name, voted))

    return votes, cropped


def detect_votes(path, metadata, write=False, write_dir=Path("./process")):
    img = cv.imread(str(path))
    cropped_image = crop_image(img)
    votes, marked_image = get_votes(cropped_image, metadata["candidates"])

    if write:
        print("\tСаналууд: ")
        for group, name, vote in votes:
            print(f"\t\t{"🟢" if vote else "🔴"} {name} [{group}]")
        print("\n")

        if not write_dir.exists():
            write_dir.mkdir()

        cv.imwrite(str(write_dir / f"{path.stem}_cropped.jpeg"), cropped_image)
        cv.imwrite(str(write_dir / f"{path.stem}_marked.jpeg"), marked_image)

    counted = len([vote for vote in votes if vote[-1]])
    if not (counted == metadata["quota"]):
        # print(
        #     f"Боловсруулалтын үр дүн ({counted}), дугуйлсан нэр дэвшигчийн тоо ({metadata["quota"]}) таарсангүй."
        # )
        logger.debug(
            f"Боловсруулалтын үр дүн ({counted}), дугуйлсан нэр дэвшигчийн тоо ({metadata["quota"]}) таарсангүй.",
            counted,
            metadata["quota"],
        )
        # raise BallotException(counted=counted, quota=metadata["quota"])
    return votes


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


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise Exception("Боловсруулалт хийх зураг эсвэл хавтсын замыг оруулна уу!")

    if len(sys.argv) >= 2 and (path := Path(sys.argv[1])) and not path.exists():
        raise FileNotFoundError("Заасан зам олдсонгүй!")

    if path.is_dir():
        filepaths = []
        for root, dirs, files in path.walk(on_error=print):
            for file in files:
                filepath = Path(root / file)
                if (
                    filepath.exists()
                    and filepath.is_file()
                    and filepath.suffix in [".jpeg", ".jpg"]
                ):
                    filepaths.append(root / file)
                else:
                    print("Алдаа: ", filepath)
        print("Боловсруулах файлын тоо: ", len(filepaths))
        metadata = get_metadata(path)
        total = len(filepaths)
        for index, path in enumerate(filepaths, start=1):
            votes = detect_votes(path, metadata, write=False)
            print(
                f"{index}/{total} Тоолсон: {len([v for v in votes if v[-1]])} Хүчинтэй: {metadata["quota"]}"
            )

    else:
        print("📄 Файл: ", path)
        metadata = get_metadata(path)
        votes = detect_votes(path, metadata, write=True)
