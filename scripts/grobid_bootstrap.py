"""
Bootstrap script to download a JRE (JDK21), fetch the Grobid binary release, and start the Grobid service locally.

Usage:
  python scripts/grobid_bootstrap.py setup   # downloads JRE + Grobid
  python scripts/grobid_bootstrap.py start   # starts Grobid on port 8070
"""

import argparse
import platform
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


# -----------------------------
# JDK auto-selection per OS/arch
# -----------------------------
def get_jdk_url():
    system = platform.system()
    machine = platform.machine().lower()

    # Windows (always x64)
    if system == "Windows":
        return "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jre_x64_windows_hotspot_21.0.4_7.zip"

    # Linux x64 (you may add aarch64 if needed)
    if system == "Linux":
        return "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jre_x64_linux_hotspot_21.0.4_7.tar.gz"

    # macOS (must choose correct architecture)
    if system == "Darwin":
        if "arm" in machine or "aarch64" in machine:
            # Apple Silicon
            return "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jre_aarch64_mac_hotspot_21.0.4_7.tar.gz"
        else:
            # Intel macOS
            return "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.4%2B7/OpenJDK21U-jre_x64_mac_hotspot_21.0.4_7.tar.gz"

    raise RuntimeError(f"Unsupported operating system for JDK download: {system}")


GROBID_ZIP_URL = "https://github.com/kermitt2/grobid/archive/refs/heads/master.zip"


# -----------------------------
# Download + extraction helpers
# -----------------------------
def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)


def extract_archive(archive: Path, target: Path):
    print(f"Extracting: {archive}")
    target.mkdir(parents=True, exist_ok=True)

    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive, "r") as z:
            z.extractall(target)
    elif archive.suffixes[-2:] == [".tar", ".gz"]:
        with tarfile.open(archive, "r:gz") as t:
            t.extractall(target)
    else:
        raise ValueError(f"Unsupported archive format: {archive}")


# -----------------------------
# Install + locate JDK
# -----------------------------
def ensure_jdk() -> Path:
    url = get_jdk_url()

    # If already downloaded, reuse
    if JDK_DIR.exists() and any(JDK_DIR.iterdir()):
        return find_java_home()

    archive = BASE_DIR / Path(url).name
    download(url, archive)
    extract_archive(archive, JDK_DIR)
    archive.unlink(missing_ok=True)

    return find_java_home()


def find_java_home() -> Path:
    if not JDK_DIR.exists():
        raise RuntimeError("JDK directory missing; run setup first.")

    for candidate in JDK_DIR.iterdir():
        if not candidate.is_dir():
            continue

        # Standard Linux/Windows layout
        std_java = candidate / "bin" / ("java.exe" if platform.system() == "Windows" else "java")
        if std_java.exists():
            return candidate.resolve()

        # macOS JDK layout
        mac_java = candidate / "Contents" / "Home" / "bin" / "java"
        if mac_java.exists():
            return (candidate / "Contents" / "Home").resolve()

    raise RuntimeError("JAVA_HOME not found. JDK extracted, but java binary is missing.")


# -----------------------------
# Install Grobid
# -----------------------------
def ensure_grobid() -> Path:
    archive = BASE_DIR / "grobid-master.zip"

    if not (GROBID_DIR.exists() and any(GROBID_DIR.iterdir())):
        print("Downloading Grobid master branch...")
        download(GROBID_ZIP_URL, archive)
        extract_archive(archive, GROBID_DIR)
        archive.unlink(missing_ok=True)

    grobid_root = (GROBID_DIR / "grobid-master").resolve()
    if not grobid_root.exists():
        raise RuntimeError("Grobid root directory missing after extraction.")

    # Make gradlew executable for macOS/Linux
    gradlew = grobid_root / "gradlew"
    if platform.system() != "Windows" and gradlew.exists():
        gradlew.chmod(gradlew.stat().st_mode | 0o111)

    return grobid_root


# -----------------------------
# Launch Grobid service
# -----------------------------
def start_grobid():
    grobid_root = ensure_grobid()
    gradlew = grobid_root / ("gradlew.bat" if platform.system() == "Windows" else "gradlew")

    if not gradlew.exists():
        raise RuntimeError(f"Gradle wrapper not found at: {gradlew}")

    java_home = ensure_jdk()

    env = os.environ.copy()
    env["JAVA_HOME"] = str(java_home)
    env["PATH"] = str(java_home / "bin") + os.pathsep + env["PATH"]

    # Platform-specific execution
    if platform.system() == "Windows":
        cmd = ["cmd.exe", "/c", str(gradlew), ":grobid-service:run"]
    else:
        cmd = [str(gradlew), ":grobid-service:run"]

    print(f"Launching Grobid on port {PORT}...")
    print("JAVA_HOME =", java_home)
    print("Executing:", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        cwd=grobid_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    time.sleep(8)
    return proc


# -----------------------------
# Main CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Bootstrap Grobid + JDK")
    parser.add_argument("action", choices=["setup", "start"])
    args = parser.parse_args()

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    if args.action == "setup":
        ensure_jdk()
        ensure_grobid()
        print("\nSetup complete! Run:")
        print("  python scripts/grobid_bootstrap.py start")
        return

    if args.action == "start":
        proc = start_grobid()
        print("Grobid started. Press Ctrl+C to stop.")
        try:
            for line in proc.stdout:
                print(line, end="")
        except KeyboardInterrupt:
            proc.terminate()


if __name__ == "__main__":
    main()
