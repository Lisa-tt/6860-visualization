"""Download and extract the official ABS 2021 GDA2020 remoteness boundaries."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


ABS_URL = (
    "https://www.abs.gov.au/statistics/standards/"
    "australian-statistical-geography-standard-asgs/"
    "edition-3-july-2021-june-2026/access-and-downloads/"
    "digital-boundary-files/RA_2021_AUST_GDA2020.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "external" / "abs"
ZIP_PATH = DESTINATION / "RA_2021_AUST_GDA2020.zip"
EXTRACTED_DIR = DESTINATION / "RA_2021_AUST_GDA2020"
SHAPEFILE = EXTRACTED_DIR / "RA_2021_AUST_GDA2020.shp"


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print("Downloading {}".format(ABS_URL))
        urlretrieve(ABS_URL, ZIP_PATH)
    else:
        print("Using existing {}".format(ZIP_PATH))

    if not SHAPEFILE.exists():
        print("Extracting {}".format(ZIP_PATH))
        with ZipFile(ZIP_PATH) as archive:
            archive.extractall(EXTRACTED_DIR)

    if not SHAPEFILE.exists():
        raise FileNotFoundError("Expected shapefile was not extracted: {}".format(SHAPEFILE))

    print("Shapefile: {}".format(SHAPEFILE))
    print("ZIP SHA256: {}".format(file_sha256(ZIP_PATH)))


if __name__ == "__main__":
    main()
