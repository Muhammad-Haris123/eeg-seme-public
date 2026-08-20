"""
Browser-assisted Dryad download for P-ADIC (AWS WAF + Anubis).

Uses local Chrome via Selenium to pass challenges, then cookies for curl/urllib
resumable downloads of large .mat files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

try:
    import requests
except ImportError:
    requests = None

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


def make_driver(download_dir: Path, headless: bool) -> webdriver.Chrome:
    download_dir.mkdir(parents=True, exist_ok=True)
    opts = Options()
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument(f"--user-agent={UA}")
    opts.add_argument("--window-size=1280,900")
    if headless:
        opts.add_argument("--headless=new")
    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    opts.add_experimental_option("prefs", prefs)
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(options=opts)
    driver.execute_cdp_cmd(
        "Page.setDownloadBehavior",
        {"behavior": "allow", "downloadPath": str(download_dir.resolve())},
    )
    return driver


def wait_past_challenges(driver: webdriver.Chrome, timeout: int = 120) -> None:
    """Wait until Anubis/WAF challenge pages clear."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        title = (driver.title or "").lower()
        src = driver.page_source[:2000].lower()
        if "validating" in title or "anubis_challenge" in src:
            time.sleep(1.5)
            continue
        if "aws waf" in src or "challenge.js" in src or "awswaf" in src:
            time.sleep(1.5)
            continue
        if "x-amzn-waf-action" in src:
            time.sleep(1.5)
            continue
        # Dryad dataset page markers
        if "dryad" in title or "p-adic" in src or "alz_c1" in src or "file_stream" in src:
            return
        time.sleep(1.0)
    raise TimeoutError(f"Challenges not cleared after {timeout}s; title={driver.title!r}")


def cookies_to_requests(driver: webdriver.Chrome) -> "requests.Session":
    if requests is None:
        raise RuntimeError("requests not installed")
    sess = requests.Session()
    sess.headers.update(
        {
            "User-Agent": UA,
            "Referer": DATASET,
            "Accept": "*/*",
        }
    )
    for c in driver.get_cookies():
        sess.cookies.set(c["name"], c["value"], domain=c.get("domain"), path=c.get("path", "/"))
    return sess


def refresh_auth(driver: webdriver.Chrome) -> None:
    driver.get(DATASET)
    wait_past_challenges(driver, timeout=180)
    # Trigger a tiny download once so WAF tokens for /downloads are issued
    driver.get(STREAM.format(fid=FILES["author"].file_id))
    wait_past_challenges(driver, timeout=180)
    # If we landed on text content, good; if download started, navigate back
    time.sleep(2)
    driver.get(DATASET)
    wait_past_challenges(driver, timeout=60)


def download_with_session(spec: FileSpec, out_dir: Path, sess: "requests.Session") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / spec.name
    url = STREAM.format(fid=spec.file_id)

    if target.exists() and target.stat().st_size == spec.size:
        print(f"[skip] {spec.name}: verifying sha256...")
        if sha256_file(target) == spec.sha256:
            print(f"[ok] {spec.name}")
            return target
        print("[warn] checksum mismatch; re-download")
        target.unlink()

    existing = target.stat().st_size if target.exists() else 0
    headers = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"
        print(f"[resume] {spec.name} @ {existing}")

    print(f"[download] {spec.name} ({spec.size/1e9:.2f} GB)")
    t0 = time.time()
    last = t0
    with sess.get(url, headers=headers, stream=True, timeout=120) as r:
        # Detect challenge HTML
        ctype = (r.headers.get("Content-Type") or "").lower()
        if r.status_code in (202, 403) or "text/html" in ctype:
            peek = r.raw.read(512)
            raise RuntimeError(
                f"blocked downloading {spec.name}: status={r.status_code} ctype={ctype} peek={peek[:120]!r}"
            )
        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        written = existing
        with target.open(mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                # bail if HTML challenge slipped through
                if written == existing and chunk[:15].lower().startswith(b"<!doctype html"):
                    raise RuntimeError("received HTML challenge body instead of file")
                f.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last >= 10:
                    speed = (written - existing) / max(now - t0, 1e-6) / 1e6
                    print(f"  {100*written/spec.size:5.1f}%  {written/1e9:.2f}/{spec.size/1e9:.2f} GB  {speed:.1f} MB/s")
                    last = now

    size = target.stat().st_size
    if size != spec.size:
        raise RuntimeError(f"size mismatch {spec.name}: {size} != {spec.size}")
    print(f"[hash] {spec.name}")
    digest = sha256_file(target)
    if digest != spec.sha256:
        raise RuntimeError(f"sha256 mismatch {spec.name}: {digest}")
    print(f"[ok] {spec.name} in {time.time()-t0:.0f}s")
    return target


def download_via_browser_click(driver: webdriver.Chrome, spec: FileSpec, out_dir: Path) -> Path:
    """Fallback: navigate to file_stream and let Chrome save the file."""
    target = out_dir / spec.name
    if target.exists() and target.stat().st_size == spec.size:
        if sha256_file(target) == spec.sha256:
            return target
        target.unlink()

    # Remove partial chrome downloads
    for p in out_dir.glob(spec.name + "*"):
        if p.suffix in {".crdownload", ".tmp"} or p.name.endswith(".crdownload"):
            try:
                p.unlink()
            except OSError:
                pass

    print(f"[browser-download] {spec.name}")
    driver.get(STREAM.format(fid=spec.file_id))
    # Wait for file to appear and finish (.crdownload gone)
    t0 = time.time()
    while time.time() - t0 < 3600 * 6:
        if target.exists() and not (out_dir / (spec.name + ".crdownload")).exists():
            if target.stat().st_size == spec.size:
                break
        # author notes may render inline as text — save page source
        if spec.name.endswith(".txt") and "AUTHOR" not in driver.title:
            body = driver.find_element(By.TAG_NAME, "body").text
            if len(body) > 100 and "Validating" not in body:
                target.write_text(body if body.endswith("\n") else body + "\n", encoding="utf-8", newline="\n")
                # may not match exact bytes; keep for inspection
                print(f"[note] saved inline text ({target.stat().st_size} bytes); sha may differ")
                return target
        time.sleep(2)
        if int(time.time() - t0) % 30 == 0:
            print(f"  waiting... exists={target.exists()} size={target.stat().st_size if target.exists() else 0}")

    if not target.exists() or target.stat().st_size != spec.size:
        raise RuntimeError(f"browser download incomplete for {spec.name}")
    digest = sha256_file(target)
    if digest != spec.sha256:
        raise RuntimeError(f"sha256 mismatch {spec.name}: {digest}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path(r"E:\padic_external"))
    ap.add_argument("--only", choices=["author", "alz", "controls", "all"], default="all")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--mode", choices=["session", "browser"], default="session")
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    driver = make_driver(out, headless=args.headless)
    try:
        print("[browser] opening Dryad + clearing challenges...")
        refresh_auth(driver)
        print("[browser] challenges cleared")

        keys = ["author", "alz", "controls"] if args.only == "all" else [args.only]
        records = []
        for key in keys:
            spec = FILES[key]
            if args.mode == "browser":
                path = download_via_browser_click(driver, spec, out)
            else:
                # re-auth before each large file
                refresh_auth(driver)
                sess = cookies_to_requests(driver)
                try:
                    path = download_with_session(spec, out, sess)
                except Exception as e:
                    print(f"[warn] session download failed ({e}); falling back to browser")
                    path = download_via_browser_click(driver, spec, out)
            records.append({**asdict(spec), "path": str(path)})

        manifest = {
            "doi": "10.5061/dryad.8gtht76pw",
            "output_dir": str(out),
            "files": records,
            "note": "AD+controls only; MCI/dep/schiz skipped",
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"[done] {out / 'manifest.json'}")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
