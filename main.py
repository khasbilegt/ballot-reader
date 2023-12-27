from pathlib import Path

import cv2 as cv
import numpy as np


def pre_process_image(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    _, threshold = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    return threshold


def find_contours(src, area_threshold=(1200, 2000), arc_threshold=(100, 200)):
    contours, _ = cv.findContours(src, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
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
    points = []
    for contour in contours:
        epsilon = 0.1 * cv.arcLength(contour, True)
        approximated_contour = cv.approxPolyDP(contour, epsilon, True)
        for point in approximated_contour.tolist():
            points.append(point[0])

    points_ndarray = np.array(points)
    hull = cv.convexHull(points_ndarray)
    approximated_hull = cv.approxPolyDP(hull, 0.1 * cv.arcLength(hull, True), True)

    top_left, top_right, bottom_left = sorted(
        [(point[0][0], point[0][1]) for point in approximated_hull.tolist()],
        key=lambda p: sum(p),
    )
    bottom_right = (top_right[0], bottom_left[1])
    return top_left, top_right, bottom_left, bottom_right


def crop_image(src):
    img = pre_process_image(src)
    contours = find_contours(img)
    top_left, top_right, bottom_left, bottom_right = calculate_edge_points(contours)

    width = max(top_right[0] - top_left[0], bottom_right[0] - bottom_left[0])
    height = max(bottom_left[1] - top_left[1], bottom_right[1] - top_right[1])

    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv.getPerspectiveTransform(pts1, pts2)
    cropped = cv.warpPerspective(src, matrix, (width, height))

    return cropped


if __name__ == "__main__":
    data = Path("./data")
    filepath = Path("2/8_974334.jpeg")

    img = cv.imread(str(data / filepath))
    cropped = crop_image(img)
    cv.imwrite("./cropped-image.jpeg", cropped)
