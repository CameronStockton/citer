"""
Bootstrap script to download a JRE (JDK21), fetch the Grobid binary release, and start the Grobid service locally.

Usage:
  python scripts/grobid_bootstrap.py setup   # downloads JRE + Grobid
  python scripts/grobid_bootstrap.py start   # starts Grobid on port 8070

This keeps everything local so users don't need to install Java manually.
"""

import argparse
import platform
import shutil
import subprocess
import tarfile
import time
import zipfile
from pathlib import Path
import os

import requests

BASE_DIR = Path("grobid_runtime")
JDK_DIR = BASE_DIR / "jdk"
GROBID_DIR = BASE_DIR / "grobid"
PORT = 8070
GROBID_VERSION = "master"

JDK_URLS = {
    "Windows": "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jre_x64_windows_hotspot_21.0.4_7.zip",
    "Linux": "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jre_x64_linux_hotspot_21.0.4_7.tar.gz",
    "Darwin": "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jre_x64_mac_hotspot_21.0.4_7.tar.gz",
}

GROBID_ZIP_URL = "https://github.com/kermitt2/grobid/archive/refs/heads/master.zip"


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def extract_archive(archive: Path, target: Path):
    target.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
    elif archive.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


def ensure_jdk() -> Path:
    system = platform.system()
    url = JDK_URLS.get(system)
    if not url:
        raise RuntimeError(f"No JDK download configured for platform {system}")
    if any(JDK_DIR.iterdir()) if JDK_DIR.exists() else False:
        return find_java_home()

    archive = BASE_DIR / Path(url).name
    print(f"Downloading JDK21 from {url}")
    download(url, archive)
    extract_archive(archive, JDK_DIR)
    archive.unlink(missing_ok=True)
    return find_java_home()


def find_java_home() -> Path:
    if not JDK_DIR.exists():
        raise RuntimeError("JDK not found; run setup first.")

    # Resolve absolute paths
    jdk_dir = JDK_DIR.resolve()

    # Search for any directory containing bin/java(.exe)
    for candidate in jdk_dir.iterdir():
        java_bin = candidate / "bin" / ("java.exe" if platform.system() == "Windows" else "java")
        if java_bin.exists():
            return candidate.resolve()

    raise RuntimeError("Valid JDK not found: missing bin/java(.exe)")



def ensure_grobid() -> Path:
    archive = BASE_DIR / "grobid-master.zip"
    extract_dest = GROBID_DIR

    if not extract_dest.exists() or not any(extract_dest.iterdir()):
        print(f"Downloading Grobid master from {GROBID_ZIP_URL}")
        download(GROBID_ZIP_URL, archive)
        extract_archive(archive, extract_dest)
        archive.unlink(missing_ok=True)

    # The extracted folder is grobid-master
    grobid_root = extract_dest / "grobid-master"
    if not grobid_root.exists():
        raise RuntimeError("Grobid extraction failed or unexpected folder structure.")

    return grobid_root.resolve()



def start_grobid():
    grobid_root = ensure_grobid()
    grobid_root = grobid_root.resolve()

    wrapper = "gradlew.bat" if platform.system() == "Windows" else "./gradlew"
    gradle_path = grobid_root / wrapper
    gradle_path = gradle_path.resolve()


    if not gradle_path.exists():
        raise RuntimeError(f"Gradle wrapper not found at {gradle_path}")

    # Set JAVA_HOME to the bundled JDK
    java_home = find_java_home()
    env = dict(**os.environ)
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str((java_home / "bin").resolve()) + os.pathsep + env["PATH"]


    if platform.system() == "Windows":
        cmd = [
            "cmd.exe",
            "/c",
            "call",
            str(gradle_path),
            ":grobid-service:run"
        ]
    else:
        cmd = [str(gradle_path), ":grobid-service:run"]


    print("Resolved grobid_root =", grobid_root)
    print("Resolved gradle_path =", gradle_path)
    print("Exists? ", gradle_path.exists())
    print("CWD =", grobid_root)
    print("ABS path =", grobid_root)
    print("JAVA_HOME set to:", java_home)
    print("Java exists:", (java_home / "bin" / "java.exe").exists())


    print(f"Starting Grobid on port {PORT} via Gradle...")
    proc = subprocess.Popen(
        cmd,
        cwd=grobid_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,       # <-- MUST include this
    )
    time.sleep(10)
    return proc



def main():
    parser = argparse.ArgumentParser(description="Bootstrap Grobid with bundled JRE")
    parser.add_argument("action", choices=["setup", "start"], help="setup downloads JRE+Grobid; start launches the service")
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.action == "setup":
        ensure_jdk()
        ensure_grobid()
        print("Setup complete. Run `python scripts/grobid_bootstrap.py start` to launch Grobid.")
    elif args.action == "start":
        ensure_jdk()
        ensure_grobid()
        proc = start_grobid()
        print("Grobid process started. Press Ctrl+C to stop.")
        try:
            for line in proc.stdout:
                print(line, end="")
        except KeyboardInterrupt:
            proc.terminate()


if __name__ == "__main__":
    main()
