import argparse
import os
import shutil
from multiprocessing import Pool
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("paths", type=str, nargs="+")
parser.add_argument("--concurrency", action="store", type=int, default=os.cpu_count())
parser.add_argument("--root", "-r", action="store_true")


def worker(path):
    filepaths = [
        Path(root, f)
        for root, _, files in path.walk(on_error=print)
        for f in files
        if Path(root, f).suffix == ".csv"
    ]
    if len(filepaths) == 1:
        filepath = filepaths[0]

        destination = Path.home() / "REPORTS" / f"{Path(filepath).parent.name}.csv"
        print(destination)
        shutil.copyfile(filepath, destination)


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
