"""Download P-ADIC via undetected-chromedriver (bypass AWS WAF / Anubis)."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests
import undetected_chromedriver as uc

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
DATASET = "https://datadryad.org/dataset/doi:10.5061/dryad.8gtht76pw"
STREAM = "https://datadryad.org/downloads/file_stream/{fid}"


@dataclass(frozen=True)
class FileSpec:
    name: str
    file_id: int
    size: int
    sha256: str


FILES = {
    "author": FileSpec(
        "AUTHOR_DATASET_SHOR_BENNINGER.txt",
        1891768,
        4374,
        "808b9cc79fb87df13d7fc0249b56e882edcc637de83cc1054ca055e027f64ee3",
    ),
    "alz": FileSpec(
        "alz_c1_new.mat",
        1891763,
        3813216803,
        "0c2fcee914d52d614596a721e385e8218017074eff9fae8a17676ba95e51576d",
    ),
    "controls": FileSpec(
        "controls_c1_new.mat",
        1891764,
        7165225913,
        "3272bb5be59f40225832d48046b85c9f79cbf343bb4328e0cf4f2d70244f5955",
    ),
}


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def wait_ready(driver, timeout: int = 240) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        title = (driver.title or "").strip()
        src = driver.page_source[:4000]
        low = src.lower()
        blocked = (
            "validating" in title.lower()
            or "anubis_challenge" in low
            or "awswaf" in low
            or "challenge.js" in low
            or "just a moment" in low
        )
        if blocked:
            time.sleep(2)
            continue
        if title or "dryad" in low or "alz_c1" in low or len(src) > 5000:
            print(f"[ready] title={title!r} elapsed={time.time()-t0:.0f}s")
            return
        time.sleep(1)
    raise TimeoutError(f"not ready; title={driver.title!r}")


def session_from_driver(driver) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "Referer": DATASET, "Accept": "*/*"})
    for c in driver.get_cookies():
        s.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))
    return s


def download(spec: FileSpec, out: Path, sess: requests.Session) -> Path:
    target = out / spec.name
    if target.exists() and target.stat().st_size == spec.size:
        if sha256_file(target) == spec.sha256:
            print(f"[ok-skip] {spec.name}")
            return target
        target.unlink()

    existing = target.stat().st_size if target.exists() else 0
    headers = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"[resume] {existing}")

    url = STREAM.format(fid=spec.file_id)
    print(f"[get] {spec.name}")
    t0 = time.time()
    last = t0
    with sess.get(url, headers=headers, stream=True, timeout=180) as r:
        ctype = (r.headers.get("Content-Type") or "").lower()
        print(f"  status={r.status_code} ctype={ctype} clen={r.headers.get('Content-Length')}")
        if r.status_code in (202, 403) or ("text/html" in ctype and r.status_code != 200):
            raise RuntimeError(f"blocked: {r.status_code} {ctype}")
        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        written = existing
        with target.open(mode) as f:
            for chunk in r.iter_content(1024 * 1024):
                if not chunk:
                    continue
                if written == existing and chunk[:15].lower().startswith(b"<!doctype"):
                    raise RuntimeError("HTML challenge body")
                f.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last >= 8:
                    print(
                        f"  {100*written/spec.size:5.1f}% "
                        f"{written/1e9:.2f}/{spec.size/1e9:.2f} GB "
                        f"{(written-existing)/max(now-t0,1e-6)/1e6:.1f} MB/s"
                    )
                    last = now
    if target.stat().st_size != spec.size:
        raise RuntimeError(f"size {target.stat().st_size} != {spec.size}")
    digest = sha256_file(target)
    if digest != spec.sha256:
        raise RuntimeError(f"sha256 {digest}")
    print(f"[ok] {spec.name}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path(r"E:\padic_external"))
    ap.add_argument("--only", choices=["author", "alz", "controls", "all"], default="author")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    opts = uc.ChromeOptions()
    opts.add_argument("--window-size=1280,900")
    prefs = {
        "download.default_directory": str(out.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
    }
    opts.add_experimental_option("prefs", prefs)

    print("[uc] launching Chrome...")
    driver = uc.Chrome(options=opts)
    try:
        driver.get(DATASET)
        wait_ready(driver, timeout=300)
        # poke download endpoint so WAF tokens exist
        driver.get(STREAM.format(fid=FILES["author"].file_id))
        time.sleep(5)
        wait_ready(driver, timeout=300)
        driver.get(DATASET)
        wait_ready(driver, timeout=120)

        sess = session_from_driver(driver)
        keys = ["author", "alz", "controls"] if args.only == "all" else [args.only]
        records = []
        for key in keys:
            # refresh cookies before large files
            driver.get(DATASET)
            wait_ready(driver, timeout=120)
            sess = session_from_driver(driver)
            path = download(FILES[key], out, sess)
            records.append({**asdict(FILES[key]), "path": str(path)})
        (out / "manifest.json").write_text(
            json.dumps(
                {"doi": "10.5061/dryad.8gtht76pw", "output_dir": str(out), "files": records},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("[done]")
    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
