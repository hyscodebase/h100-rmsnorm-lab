"""Prepare the pinned official CUDA operator library without altering installed packages."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

WHEEL = "vllm-0.10.0-cp38-abi3-manylinux1_x86_64.whl"
WHEEL_HASH = "8ca37559d82b43b5e8c8248d2e4a1ecb51d6d4e5d517491d656df6491ed93dab"
WHEEL_BYTES = 386_575_356
URL = "https://files.pythonhosted.org/packages/a2/98/a9c7fd61e27fd766f3cb7022613b214f053836ba7749d12a4cd5ab4644fc/" + WHEEL
MEMBER = "vllm/_C.abi3.so"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stage(stream, target, limit, expected=None):
    digest = hashlib.sha256()
    size = 0
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as tmp:
        temp = Path(tmp.name)
        try:
            while chunk := stream.read(4 * 2**20):
                size += len(chunk)
                if size > limit:
                    raise ValueError("Downloaded asset exceeds the registered size.")
                tmp.write(chunk)
                digest.update(chunk)
            if size != limit or (expected is not None and expected != digest.hexdigest()):
                raise ValueError("Downloaded bytes differ from the registered release.")
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
    return digest.hexdigest()


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=True)
    wheel, lib = directory / WHEEL, directory / Path(MEMBER).name
    if wheel.is_file():
        if wheel.stat().st_size != WHEEL_BYTES or sha(wheel) != WHEEL_HASH:
            raise ValueError("The existing wheel differs from the official release.")
    else:
        if shutil.disk_usage(directory).free < 2 * WHEEL_BYTES:
            raise ValueError("Insufficient storage for the isolated wheel and extension.")
        with urllib.request.urlopen(URL, timeout=30) as download:
            stage(download, wheel, WHEEL_BYTES, WHEEL_HASH)
    with zipfile.ZipFile(wheel) as archive, archive.open(MEMBER) as stream:
        info = archive.getinfo(MEMBER)
        if lib.is_file():
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
            if sha(lib) != expected:
                raise ValueError("Existing CUDA extension differs from the verified wheel.")
        else:
            expected = stage(stream, lib, info.file_size)
    return {"version": "0.10.0", "wheel": WHEEL, "wheel_sha256": WHEEL_HASH,
            "extension": Path(MEMBER).name, "extension_sha256": expected,
            "extension_size_bytes": lib.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.directory), indent=2))
