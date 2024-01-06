import argparse
import codecs
import csv
import logging
import os
from multiprocessing import Pool
from pathlib import Path

import cv2 as cv
import numpy as np
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("paths", type=str, nargs="+")
parser.add_argument("--concurrency", action="store", type=int, default=os.cpu_count())
parser.add_argument("--root", "-r", action="store_true")
parser.add_argument("--metadata", "-m", action="store_true")


def correct_path(path: str):
    source_path = "/Volumes/New Volume/2020 УИХ сонгууль зурган файл/"

    if (prefix := "/Users/oneline/Documents/ballots/") and path.startswith(prefix):
        corrected_path = path.replace(prefix, source_path)
    elif (prefix := "/Users/khasbilegt/INPUT/") and path.startswith(prefix):
        corrected_path = path.replace(prefix, source_path)
    else:
        corrected_path = path
    return corrected_path, corrected_path.removeprefix(source_path)


def pre_process_image(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    return threshold


def find_contours(img):
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
    ordered_contours = []
    for contour in contours:
        points = np.concatenate(contour)
        x_ordered = sorted(points, key=lambda x: x[0])

        left_top, left_bottom = sorted(x_ordered[:2], key=lambda x: x[1])
        right_top, right_bottom = sorted(x_ordered[2:], key=lambda x: x[1])
        ordered_contours.append((left_top, right_top, left_bottom, right_bottom))

    top_left = ordered_contours[0][2]
    top_right = ordered_contours[1][3]
    bottom_left = ordered_contours[2][2]
    bottom_right = [top_right[0], bottom_left[-1]]
    return top_left, top_right, bottom_left, bottom_right


def crop_image(img):
    image = pre_process_image(img)
    contours = find_contours(image)
    top_left, top_right, bottom_left, bottom_right = calculate_edge_points(contours)

    width = top_right[0] - top_left[0]
    height = bottom_left[1] - top_left[1]

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
        y_start = 20 + y_weight
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


def detect_votes(path: Path, metadata, processed_path):
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
            cv.imwrite(str(Path(processed_path, path.name)), converted_image)

        return votes, vote_count
    except Exception as e:
        print(path, e)
        raise e


def get_metadata(path, metadata_path, row_offset=9, column_offset=11):
    if (path := Path(metadata_path, f"{path.stem}.yaml")) and not path.exists():
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


def worker(path):
    print("📁 Хавтас: " if path.is_dir() else "📄 Файл: ", path)
    if path.is_dir():
        output_path = Path.home() / "OUTPUT" / path.stem
        metadata_path = Path.home() / "METADATA"

        if (
            metadata_file_path := Path(metadata_path, f"{path.stem}.yaml")
        ) and not metadata_file_path.exists():
            print(f"{metadata_file_path} does not exist")
            return

        if not output_path.exists():
            output_path.mkdir(parents=True)

        logging.basicConfig(
            filename=Path(output_path, "election.log"),
            filemode="w",
            format="[%(asctime)s] %(levelname)s - %(message)s",
        )

        filepaths = sorted(
            [
                Path(root, f)
                for root, _, files in path.walk(on_error=print)
                for f in files
                if Path(root, f).suffix in [".jpeg", ".jpg"]
            ]
        )
        total = len(filepaths)
        metadata = get_metadata(path, metadata_path)
        with Path(output_path, "report.csv").open(mode="w") as report:
            report.write(codecs.BOM_UTF8.decode())
            report_writer = csv.writer(report)

            report_writer.writerow(
                ["Дугаар", "Тойрог", "Хэсгийн хороо"]
                + [
                    f"{name} ({group_name})"
                    for group_name, name, _, _ in metadata["candidates"]
                ]
                + ["Тоолсон", "Дугайлах", "Зөв", "Зам"]
            )

            for index, filepath in enumerate(filepaths, start=1):
                try:
                    votes, count = detect_votes(filepath, metadata, output_path)
                    print(
                        f"{index}/{total} - {count}({metadata["quota"]}) - {filepath}"
                    )

                    corrected_path, normalized_path = correct_path(str(filepath))
                    segments = normalized_path.split("/")
                    toirog = segments[0]

                    if (
                        toirog.startswith("7")
                        or toirog.startswith("9")
                        or toirog.startswith("26")
                    ):
                        khoroo = [toirog.split(" ", maxsplit=1)[0], segments[2]]
                    else:
                        khoroo = segments[1].split("-", maxsplit=1)

                    report_writer.writerow(
                        [index]
                        + khoroo
                        + [int(vote) for _, _, vote in votes]
                        + [
                            count,
                            metadata["quota"],
                            count == metadata["quota"],
                            corrected_path,
                        ]
                    )
                except Exception as e:
                    logging.error(f"{filepath} - {str(e)}")
                    report_writer.writerow(
                        [index]
                        + [0 for _ in range(int(metadata["quota"]))]
                        + [
                            0,
                            metadata["quota"],
                            False,
                            filepath,
                        ]
                    )
                    continue
    else:
        metadata = get_metadata(path)
        votes, count = detect_votes(path, metadata)
        candidate_names = [f"{v[1]} ({v[0]})" for v in votes if v[-1]]
        print(
            f"\tТоолсон: {count} ({", ".join(candidate_names)})\n\tХүчинтэй: {metadata["quota"]}",
        )


if __name__ == "__main__":
    args = parser.parse_args()

    if args.root:
        if len(args.paths) != 1:
            raise RuntimeError("ddd")

        root = args.paths[0]
        try:
            paths = [
                Path(root, i)
                for i in os.listdir(root)
                if not i.startswith(".") and os.path.isdir(Path(root, i))
            ]
        except NotADirectoryError:
            print(f"{root} is not a directory")
            exit(1)
    else:
        paths = args.paths

    with Pool(processes=min(args.concurrency, len(paths))) as pool:
        pool.map(worker, (Path(i) for i in paths))
