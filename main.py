from pathlib import Path

import cv2 as cv
import numpy as np


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
    return whites > blacks


def get_votes(img, template, x_weight=0, y_weight=0):
    votes = []
    cropped = np.array(img, copy=True)
    for x, y, candidate in template:
        x_start = 66 + x_weight
        x_step = 11
        y_start = 54 + y_weight
        y_step = 12

        left = x_start + 2 * x_step * (x - 1)
        right = left + x_step * 2
        top = y_start + 2 * y_step * (y - 1)
        bottom = top + y_step

        cv.rectangle(cropped, [left, top], [right, bottom], (255, 0, 0), 2)

        voted = is_filled(img[top:bottom, left:right])
        votes.append((candidate, voted))
    return votes, cropped


def detect_votes(path, template, write=False, write_dir=Path("./process")):
    img = cv.imread(str(path))
    cropped_image = crop_image(img)
    votes, marked_image = get_votes(cropped_image, template)

    if write:
        if not write_dir.exists():
            write_dir.mkdir()

        cv.imwrite(str(write_dir / f"{path.stem}_cropped.jpeg"), cropped_image)
        cv.imwrite(str(write_dir / f"{path.stem}_marked.jpeg"), marked_image)
    return votes


if __name__ == "__main__":
    data = Path("./data")
    filepath = Path("5/1807.jpeg")

    votes = detect_votes(
        data / filepath,
        template=[
            (1, 10, "Бямбасүрэнгийн ЭНХ-АМГАЛАН"),
            (1, 11, "Шатарбалын РАДНААСЭД"),
            (12, 10, "Очирбатын АМГАЛАНБААТАР"),
            (12, 11, "Осормаагийн БАТХАНД"),
            (23, 10, "Базаррагчаагийн ОЮУНБИЛЭГ"),
            (23, 11, "Сандуйн БАТБААТАР"),
        ],
        write=True,
    )
    print(votes)
