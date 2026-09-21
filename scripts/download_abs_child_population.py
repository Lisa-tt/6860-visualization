"""Download the ABS 2021 GCP Remoteness Area child-population DataPack."""

from hashlib import sha256
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile


ABS_URL = (
    "https://www.abs.gov.au/census/find-census-data/datapacks/download/"
    "2021_GCP_RA_for_AUS_short-header.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "data" / "external" / "abs"
ZIP_PATH = DESTINATION / "2021_GCP_RA_for_AUS_short-header.zip"
EXPECTED_MEMBER = (
    "2021 Census GCP Remoteness Areas for AUS/"
    "2021Census_G04A_AUST_RA.csv"
)


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading {ABS_URL}")
        urlretrieve(ABS_URL, ZIP_PATH)
    else:
        print(f"Using existing {ZIP_PATH}")

    with ZipFile(ZIP_PATH) as archive:
        if EXPECTED_MEMBER not in archive.namelist():
            raise FileNotFoundError(
                f"Expected G04A table was not found in DataPack: {EXPECTED_MEMBER}"
            )

    print(f"DataPack: {ZIP_PATH}")
    print(f"ZIP SHA256: {file_sha256(ZIP_PATH)}")


if __name__ == "__main__":
    main()
