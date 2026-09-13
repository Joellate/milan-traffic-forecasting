"""Builds a date -> (zip_path, entry_name) manifest from the downloaded
Harvard Dataverse zip parts, de-duplicating dates that appear in more than
one zip (the downloader split overlapping day ranges across parts)."""
import re
import zipfile
from pathlib import Path

DOWNLOADS_DIR = Path(r"C:\Users\AltTronic\Downloads")
ZIP_NAMES = [
    "dataverse_files.zip",
    "dataverse_files (1).zip",
    "dataverse_files (2).zip",
    "dataverse_files (3).zip",
    "dataverse_files (4).zip",
    "dataverse_files (5).zip",
    "dataverse_files (6).zip",
]
FNAME_RE = re.compile(r"sms-call-internet-mi-(\d{4}-\d{2}-\d{2})\.txt")


def build_manifest():
    manifest = {}
    for name in ZIP_NAMES:
        zpath = DOWNLOADS_DIR / name
        if not zpath.exists():
            continue
        with zipfile.ZipFile(zpath) as zf:
            for entry in zf.namelist():
                m = FNAME_RE.match(entry)
                if not m:
                    continue
                date = m.group(1)
                if date not in manifest:
                    manifest[date] = (str(zpath), entry)
    return dict(sorted(manifest.items()))


if __name__ == "__main__":
    manifest = build_manifest()
    print(f"{len(manifest)} unique daily files found")
    dates = list(manifest.keys())
    print("range:", dates[0], "to", dates[-1])
