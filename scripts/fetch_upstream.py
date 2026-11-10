"""Fetch the exact published source snapshot without installing it."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "c5a45afa9f1541a33e7034271b75b64fa8291526"
REPOSITORY = "dsz532/exeGen"


def main():
    destination = ROOT / "third_party" / "exegen"
    with urlopen(
        f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1",
        timeout=60,
    ) as response:
        tree = json.load(response)
    manifest = {"repository": REPOSITORY, "revision": REVISION, "files": {}}
    for entry in tree["tree"]:
        if entry["type"] != "blob":
            continue
        relative = entry["path"]
        with urlopen(
            f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{relative.replace(' ', '%20')}",
            timeout=60,
        ) as response:
            content = response.read()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["files"][relative] = hashlib.sha256(content).hexdigest()
    (destination.parent / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(manifest['files'])} upstream files at {REVISION}")


if __name__ == "__main__":
    main()
