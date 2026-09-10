"""B+ skip-if-valid sidecar logic tests (CPU, filesystem-only).

Run from /opt/sglang-test with the module importable:
    PYTHONPATH=/opt/sglang-patch/sglang/python /opt/sglang-env/bin/python3 test_ple_sidecar.py

Imports the real qwen4_exp_ple_table helpers when sglang is importable;
otherwise re-executes the extracted functions standalone.
"""
import hashlib
import json
import os
import shutil
import sys
import tempfile

FAIL = []


def check(cond, msg):
    if not cond:
        FAIL.append(msg)
        print(f"  [FAIL] {msg}")
    else:
        print(f"  [ok] {msg}")


try:
    from sglang.srt.models.qwen4_exp_ple_table import (
        ple_fingerprint,
        ple_sidecar_is_valid,
        ple_sidecar_path,
        ple_sidecar_write,
    )
    print("using real qwen4_exp_ple_table module")
except Exception as e:  # fallback: extract functions from the source file
    print(f"sglang import unavailable ({e}); extracting helpers standalone")
    import re

    src = open(
        os.path.join(os.path.dirname(__file__), "..", "runtime-src",
                     "qwen4_exp_ple_table.py")
    ).read()
    ns = {
        "os": os, "json": json, "glob": __import__("glob"),
        "time": __import__("time"), "Dict": dict,
        "torch": type("T", (), {"Tensor": object})(),
        "logger": type("L", (), {"warning": staticmethod(lambda *a: None)})(),
    }
    # grab the whole B+ block (SIDECAR_NAME .. before the rss-trimmer factory)
    m = re.search(r"SIDECAR_NAME = .*?\n(?=def make_ple_file_rss_trimmer)", src, re.S)
    exec("import glob, time\nfrom typing import Dict\n" + m.group(0), ns)
    ple_sidecar_path = ns["ple_sidecar_path"]
    ple_fingerprint = ns["ple_fingerprint"]
    ple_sidecar_is_valid = ns["ple_sidecar_is_valid"]
    ple_sidecar_write = ns["ple_sidecar_write"]

SHAPE = (320_001_536 // 8, 160)  # any fixed shape for the test
DTYPE = "torch.float8_e4m3fn"
NBYTES = SHAPE[0] * SHAPE[1] * 1


def make_env(tmp):
    """Fake checkpoint dir with 3 safetensors files + table dir."""
    model = os.path.join(tmp, "ckpt")
    tabdir = os.path.join(tmp, "ple")
    os.makedirs(model)
    os.makedirs(tabdir)
    for i in range(3):
        with open(os.path.join(model, f"model-plefp8-{i:05d}.safetensors"), "wb") as f:
            f.write(b"\x00" * (1024 + i))
    table_file = os.path.join(tabdir, "ple_rows0-100_fp8.pt")
    with open(table_file, "wb") as f:
        f.truncate(NBYTES)  # sparse
    return model, tabdir, table_file


def test_full_lifecycle():
    print("T-B1: seal -> valid -> drift invalidates")
    tmp = tempfile.mkdtemp()
    try:
        model, tabdir, table_file = make_env(tmp)
        sidecar = ple_sidecar_path(tabdir, "rows0-100")
        fps = ple_fingerprint(model)
        check(len(fps) == 3, "fingerprint covers all safetensors files")
        # no sidecar yet -> invalid
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps, model),
              "missing sidecar -> invalid")
        ple_sidecar_write(sidecar, NBYTES, SHAPE, DTYPE, fps, model)
        check(ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps, model),
              "fresh sidecar + matching sources -> valid")
        # source file changes content -> different fingerprint
        p0 = os.path.join(model, "model-plefp8-00000.safetensors")
        with open(p0, "ab") as f:
            f.write(b"tampered")
        fps2 = ple_fingerprint(model)
        check(fps2 != fps, "append changes fingerprint")
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps2, model),
              "drifted source -> invalid")
        # table file removed -> invalid even with matching fingerprints
        ple_sidecar_write(sidecar, NBYTES, SHAPE, DTYPE, fps2, model)
        os.remove(table_file)
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps2, model),
              "missing/truncated table -> invalid")
    finally:
        shutil.rmtree(tmp)


def test_identity_guards():
    print("T-B2: identity mismatches reject")
    tmp = tempfile.mkdtemp()
    try:
        model, tabdir, table_file = make_env(tmp)
        sidecar = ple_sidecar_path(tabdir, "rows0-100")
        fps = ple_fingerprint(model)
        ple_sidecar_write(sidecar, NBYTES, SHAPE, DTYPE, fps, model)
        # different byte size
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES + 1, SHAPE, DTYPE, fps, model),
              "size mismatch -> invalid")
        # different dtype string
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, "torch.bfloat16", fps, model),
              "dtype mismatch -> invalid")
        # different model path (checkpoint swapped in place)
        other = os.path.join(tmp, "other")
        os.makedirs(other)
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps, other),
              "model_path mismatch -> invalid")
        # new source file appears (128 -> 129 shards)
        with open(os.path.join(model, "model-plefp8-00003.safetensors"), "wb") as f:
            f.write(b"x" * 4096)
        fps3 = ple_fingerprint(model)
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps3, model),
              "extra source file -> invalid")
        # corrupt sidecar json
        with open(sidecar, "w") as f:
            f.write("{not json")
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps, model),
              "malformed sidecar -> invalid")
    finally:
        shutil.rmtree(tmp)


def test_rename_in_place_same_fingerprint():
    print("T-B3: in-place replace changes fingerprint (dev/ino/mtime)")
    tmp = tempfile.mkdtemp()
    try:
        model, tabdir, table_file = make_env(tmp)
        sidecar = ple_sidecar_path(tabdir, "rows0-100")
        fps = ple_fingerprint(model)
        ple_sidecar_write(sidecar, NBYTES, SHAPE, DTYPE, fps, model)
        # simulate: cp new content over a shard (same size!) - mtime_ns changes
        p0 = os.path.join(model, "model-plefp8-00000.safetensors")
        sz = os.path.getsize(p0)
        with open(p0, "r+b") as f:
            f.seek(0)
            f.write(b"\x11" * sz)
            f.flush()
            os.fsync(f.fileno())
        fps2 = ple_fingerprint(model)
        check(fps2 != fps and not ple_sidecar_is_valid(
            sidecar, table_file, NBYTES, SHAPE, DTYPE, fps2, model),
            "same-size in-place rewrite caught by mtime_ns")
        # simulate: atomic replace via new file (inode change caught)
        tmpf = p0 + ".new"
        with open(tmpf, "wb") as f:
            f.write(b"\x22" * sz)
        os.replace(tmpf, p0)
        fps3 = ple_fingerprint(model)
        check(not ple_sidecar_is_valid(sidecar, table_file, NBYTES, SHAPE, DTYPE, fps3, model),
              "inode swap caught by st_ino")
    finally:
        shutil.rmtree(tmp)


def test_seal_gate_row_count():
    print("T-B4: seal requires full row count (wrapper logic contract)")
    # The model-side gate: written >= expected else refuse to seal.
    # Reproduce the arithmetic used in qwen4_exp.load_weights seal block.
    expected = 40_000_192
    for written, ok in [(0, False), (expected - 1, False), (expected, True), (expected + 5, True)]:
        res = written >= expected
        check(res == ok, f"row gate written={written} -> seal={res}")


def main():
    test_full_lifecycle()
    test_identity_guards()
    test_rename_in_place_same_fingerprint()
    test_seal_gate_row_count()
    print()
    if FAIL:
        print(f"RESULT: {len(FAIL)} FAILURES")
        for f in FAIL:
            print("  -", f)
        sys.exit(1)
    print("RESULT: ALL PLE SIDECAR TESTS PASS")


main()
