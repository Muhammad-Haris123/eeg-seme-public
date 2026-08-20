"""
Download P-ADIC Dryad files (AD + controls only), bypassing Anubis PoW.

DOI: 10.5061/dryad.8gtht76pw
URL pattern (from transportability study): datadryad.org/downloads/file_stream/{id}
"""
from __future__ import annotations

import argparse
import hashlib
import http.cookiejar
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
BASE = "https://datadryad.org"


@dataclass(frozen=True)
class FileSpec:
    name: str
    file_id: int
    size: int
    sha256: str


# Hashes from noronhareuben1/eeg-slowing-transportability
FILES = {
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
    "author": FileSpec(
        "AUTHOR_DATASET_SHOR_BENNINGER.txt",
        1891768,
        4374,
        "808b9cc79fb87df13d7fc0249b56e882edcc637de83cc1054ca055e027f64ee3",
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


def solve_pow(random_data: str, difficulty: int) -> tuple[int, str]:
    target = "0" * difficulty
    nonce = 0
    while True:
        digest = hashlib.sha256(f"{random_data}{nonce}".encode()).hexdigest()
        if digest[:difficulty] == target:
            return nonce, digest
        nonce += 1


def parse_challenge(html: str) -> dict | None:
    m = re.search(
        r'<script id="anubis_challenge"[^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    if not m:
        return None
    return json.loads(m.group(1))


def opener_with_cookies(jar: http.cookiejar.CookieJar):
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def ensure_anubis_cookie(jar: http.cookiejar.CookieJar, seed_url: str) -> None:
    """Fetch a page, solve Anubis if present, store JWT cookie."""
    op = opener_with_cookies(jar)
    req = urllib.request.Request(seed_url, headers={"User-Agent": UA, "Referer": BASE + "/"})
    with op.open(req, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="replace")
        final_url = resp.geturl()

    data = parse_challenge(html)
    if data is None:
        print("[anubis] no challenge (already authorized or unprotected)")
        return

    algorithm = data["rules"]["algorithm"]
    difficulty = data["rules"]["difficulty"]
    random_data = data["challenge"]["randomData"]
    challenge_id = data["challenge"]["id"]
    print(f"[anubis] solving {algorithm} difficulty={difficulty} ...")
    t0 = time.time()
    if algorithm != "fast":
        raise RuntimeError(f"unsupported Anubis algorithm: {algorithm}")
    nonce, digest = solve_pow(random_data, difficulty)
    elapsed_ms = int((time.time() - t0) * 1000)
    print(f"[anubis] solved nonce={nonce} in {elapsed_ms} ms")

    params = urllib.parse.urlencode(
        {
            "id": challenge_id,
            "response": digest,
            "nonce": nonce,
            "redir": final_url,
            "elapsedTime": elapsed_ms,
        }
    )
    submit = f"{BASE}/.within.website/x/cmd/anubis/api/pass-challenge?{params}"

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    op2 = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(jar),
        NoRedirect(),
    )
    req2 = urllib.request.Request(submit, headers={"User-Agent": UA})
    try:
        op2.open(req2, timeout=60)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f"pass-challenge failed: HTTP {e.code}") from e

    names = [c.name for c in jar]
    if not any("anubis" in n.lower() for n in names):
        raise RuntimeError(f"Anubis cookie missing; got {names}")
    print(f"[anubis] cookies: {names}")


def download_file(spec: FileSpec, out_dir: Path, jar: http.cookiejar.CookieJar) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / spec.name
    url = f"{BASE}/downloads/file_stream/{spec.file_id}"

    if target.exists() and target.stat().st_size == spec.size:
        print(f"[skip] {spec.name}: size match, verifying sha256...")
        if sha256_file(target) == spec.sha256:
            print(f"[ok] {spec.name}")
            return target
        print(f"[warn] checksum mismatch; re-downloading {spec.name}")
        target.unlink()

    # resume if partial
    existing = target.stat().st_size if target.exists() else 0
    headers = {
        "User-Agent": UA,
        "Referer": f"{BASE}/dataset/doi:10.5061/dryad.8gtht76pw",
        "Accept": "*/*",
    }
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        print(f"[resume] {spec.name} from byte {existing}")

    # refresh Anubis cookie before large transfer
    ensure_anubis_cookie(jar, f"{BASE}/dataset/doi:10.5061/dryad.8gtht76pw")
    op = opener_with_cookies(jar)
    req = urllib.request.Request(url, headers=headers)
    print(f"[download] {spec.name} ({spec.size / 1e9:.2f} GB) -> {target}")
    t0 = time.time()
    last_log = t0
    with op.open(req, timeout=120) as resp:
        mode = "ab" if existing and resp.status == 206 else "wb"
        if mode == "wb" and existing:
            # server ignored Range — restart
            existing = 0
        written = existing
        with target.open(mode) as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                now = time.time()
                if now - last_log >= 10:
                    mb_s = (written - existing) / max(now - t0, 1e-6) / 1e6
                    pct = 100.0 * written / spec.size
                    print(f"  {pct:5.1f}%  {written/1e9:.2f}/{spec.size/1e9:.2f} GB  {mb_s:.1f} MB/s")
                    last_log = now

    if target.stat().st_size != spec.size:
        raise RuntimeError(
            f"size mismatch for {spec.name}: got {target.stat().st_size}, want {spec.size}"
        )
    print(f"[hash] verifying {spec.name}...")
    digest = sha256_file(target)
    if digest != spec.sha256:
        raise RuntimeError(f"sha256 mismatch for {spec.name}: {digest}")
    print(f"[ok] {spec.name} in {time.time() - t0:.0f}s")
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\padic_external"),
        help="Output directory (default E:\\padic_external due to C: free space)",
    )
    ap.add_argument(
        "--only",
        choices=["author", "alz", "controls", "all"],
        default="all",
    )
    args = ap.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    jar = http.cookiejar.CookieJar()
    ensure_anubis_cookie(jar, f"{BASE}/dataset/doi:10.5061/dryad.8gtht76pw")

    order = ["author", "alz", "controls"] if args.only == "all" else [args.only]
    records = []
    for key in order:
        path = download_file(FILES[key], out, jar)
        records.append({**asdict(FILES[key]), "path": str(path)})

    manifest = {
        "doi": "10.5061/dryad.8gtht76pw",
        "version_date": "2022-10-27",
        "output_dir": str(out),
        "files": records,
        "note": "AD+controls only; MCI/dep/schiz skipped by design",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[done] manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
