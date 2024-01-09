import argparse
import codecs
import csv
import os
from multiprocessing import Pool
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("paths", type=str, nargs="+")
parser.add_argument("--concurrency", action="store", type=int, default=os.cpu_count())
parser.add_argument("--root", "-r", action="store_true")


def correct_path(path: str):
    source_path = "/Volumes/New Volume/2020 УИХ сонгууль зурган файл/"

    if (prefix := "/Users/oneline/Documents/ballots/") and path.startswith(prefix):
        corrected_path = path.replace(prefix, source_path)
    elif (prefix := "/Users/khasbilegt/INPUT/") and path.startswith(prefix):
        corrected_path = path.replace(prefix, source_path)
    else:
        corrected_path = path
    return corrected_path, corrected_path.removeprefix(source_path)


def worker(path: Path):
    print("📄", path.name)
    converted_path = path.parent.parent / "converted_reports" / path.name

    with path.open("r") as report:
        with converted_path.open("w") as converted:
            converted.write(codecs.BOM_UTF8.decode())

            csv_reader = csv.reader(report)
            csv_writer = csv.writer(converted)
            entries = list(csv_reader)
            header_row = entries[0]
            csv_writer.writerow(
                header_row[:1] + ["Тойрог", "Хэсгийн хороо"] + header_row[1:]
            )
            for row in entries[1:]:
                image_path = row[-1]
                corrected_path, normalized_path = correct_path(image_path)

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

                csv_writer.writerow(row[:1] + khoroo + row[1:-1] + [corrected_path])


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
                if not str(i).startswith(".")
                and str(i).endswith(".csv")
                and os.path.isfile(Path(root, i))
            ]
        except NotADirectoryError:
            print(f"{root} is not a directory")
            exit(1)
    else:
        paths = args.paths

    with Pool(processes=min(args.concurrency, len(paths))) as pool:
        pool.map(worker, (Path(i) for i in paths))
