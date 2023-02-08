import os
import pandas as pd
import shutil
import pathlib
import tqdm

def remake_folders() -> None:
    try:
        os.mkdir(os.path.join(os.getcwd(), "Facebook"))
    except FileExistsError:
        shutil.rmtree(os.path.join(os.getcwd(), "Facebook"))
        os.mkdir(os.path.join(os.getcwd(), "Facebook"))
    try:
        os.mkdir(os.path.join(os.getcwd(), "Instagram"))
    except FileExistsError:
        shutil.rmtree(os.path.join(os.getcwd(), "Instagram"))
        os.mkdir(os.path.join(os.getcwd(), "Instagram"))
    try:
        os.mkdir(os.path.join(os.getcwd(), "Twitter"))
    except FileExistsError:
        shutil.rmtree(os.path.join(os.getcwd(), "Twitter"))
        os.mkdir(os.path.join(os.getcwd(), "Twitter"))

def get_platform(c: str):
    if c.upper() == "F":
        return "Facebook"
    elif c.upper() == "I":
        return "Instagram"
    elif c.upper() == "T":
        return "Twitter"
    else:
        raise Exception("Unknown platform")
    
def move_processed_data():
    files = list(os.listdir("processed"))
    for f in tqdm.tqdm(files):
        platform_folder = get_platform(f[0])
        df = pd.read_csv(os.path.join("processed", f), header=None, engine="python")
        df.to_csv(
        f"{platform_folder}/{f}", index=False, header=False
        )

def recursive_files_list(src):
    root = pathlib.Path(src)
    full_paths = list(root.rglob("*"))
    files = []
    for path in full_paths:
        files.append(str(path))
    # files[:] = [x for x in files if not x.isdigit()]
    files[:] = [x for x in files if not x.startswith('.')]
    files[:] = [x for x in files if x.endswith('fpd1.csv')]
    print("Length:", len(files))
    return files
remake_folders()

files = recursive_files_list(os.path.join(os.getcwd(), 'raw_data'))
for file in files:
    try:
          df = pd.read_csv(file, header=None, engine="python")
          new_filename = os.path.basename(file)[:-9] + ".csv"
          platform_folder = get_platform(new_filename[0])
          print(new_filename)
          df.to_csv(
        f"{platform_folder}/{new_filename}", index=False, header=False
    )
    except ValueError:
          print("Error on file:", file)
          input("HELLO")
print(">>>>>>Moving pre-processed files")
move_processed_data()