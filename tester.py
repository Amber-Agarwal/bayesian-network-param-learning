#!/usr/bin/env python3
import os
import sys
import subprocess
import time
import re
import shutil
from pathlib import Path
from datetime import datetime

# ===================== CONFIG (expected names in work dir) =====================
BIF_TARGET_NAME   = "hailfinder.bif"
DAT_TARGET_NAME   = "records.dat"
GOLD_TARGET_NAME  = "gold_hailfinder.bif"
SOLVED_FILE_NAME  = "solved.bif"

WORK_ROOT   = Path("work")
OUTPUT_ROOT = Path("test_case_checker/output")
TC_ROOT     = Path("test_case_checker/tc")
# ==============================================================================

# ANSI colors
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def print_banner():
    print(f"""
{Colors.CYAN}{Colors.BOLD}╔═══════════════════════════════════════════════════════════╗
║        BAYESIAN NETWORK TEST RUNNER (exact EM harness)    ║
║        Weather Forecasting Assignment                     ║
╚═══════════════════════════════════════════════════════════╝{Colors.ENDC}
""")

def print_section(title):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*60}{Colors.ENDC}\n")

# -------- helper: robust tc file picking --------
def _pick_file(tc_dir: Path, patterns):
    for pat in patterns:
        hits = sorted(tc_dir.glob(pat))
        if hits:
            return hits[0]
    return None

def find_test_cases():
    if not TC_ROOT.exists():
        print(f"{Colors.RED}Error: {TC_ROOT} not found!{Colors.ENDC}")
        return []
    tcs = [d for d in TC_ROOT.iterdir() if d.is_dir()]
    return sorted(tcs, key=lambda p: p.name)

def display_test_cases(tcs):
    print(f"{Colors.CYAN}{Colors.BOLD}Available Test Cases:{Colors.ENDC}\n")
    for i, tc in enumerate(tcs, 1):
        # try to find files with forgiving patterns
        bif  = _pick_file(tc, ["*hailfinder*.bif", "*.bif"])
        dat  = _pick_file(tc, ["records*.dat", "*.dat"])
        gold = _pick_file(tc, ["gold*hailfinder*.bif", "gold*.bif"])
        ok = bool(bif and dat and gold)
        status = f"{Colors.GREEN}✓{Colors.ENDC}" if ok else f"{Colors.RED}✗{Colors.ENDC}"
        print(f"  {Colors.BOLD}[{i}]{Colors.ENDC} {status} {Colors.CYAN}{tc.name}{Colors.ENDC}")
        if bif:  print(f"      └─ BIF:  {bif.name}")
        if dat:  print(f"      └─ DAT:  {dat.name}")
        if gold: print(f"      └─ GOLD: {gold.name}")
        print()

def select_test_cases(tcs):
    while True:
        s = input(f"{Colors.YELLOW}Enter test case numbers (comma-separated) or 'all': {Colors.ENDC}").strip().lower()
        if s == "all":
            return tcs
        try:
            idxs = [int(x.strip()) for x in s.split(",")]
            sel = [tcs[i-1] for i in idxs if 1 <= i <= len(tcs)]
            if sel:
                return sel
        except Exception:
            pass
        print(f"{Colors.RED}Invalid selection. Try again.{Colors.ENDC}")

def run_compile():
    """
    Compile without relying on bash. Prefer g++, then clang++.
    Produces ./start and ./format (or .exe on Windows).
    """
    print(f"{Colors.YELLOW}Compiling C++ sources...{Colors.ENDC}")

    def have(prog):
        try:
            subprocess.run([prog, "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            return True
        except Exception:
            return False

    compiler = None
    for cand in ("g++", "clang++"):
        if have(cand):
            compiler = cand
            break

    if compiler is None:
        # fallback: try bash compile.sh if available
        print(f"{Colors.YELLOW}No g++/clang++ on PATH; trying bash compile.sh...{Colors.ENDC}")
        try:
            res = subprocess.run(["bash", "compile.sh"], text=True, capture_output=True, timeout=300)
            if res.returncode == 0:
                print(f"{Colors.GREEN}✓ Compilation successful via bash compile.sh{Colors.ENDC}")
                return True
            print(f"{Colors.RED}✗ Compilation failed via bash compile.sh{Colors.ENDC}")
            if res.stdout.strip(): print(res.stdout.strip())
            if res.stderr.strip(): print(res.stderr.strip())
            return False
        except Exception as e:
            print(f"{Colors.RED}✗ Could not run bash compile.sh: {e}{Colors.ENDC}")
            return False

    # Direct compile path
    print(f"{Colors.CYAN}Using compiler: {compiler}{Colors.ENDC}")
    cmds = [
        [compiler, "-O3", "-std=c++17", "startup_code.cpp", "-o", "start"],
        [compiler, "-O3", "-std=c++17", "Format_Checker.cpp",          "-o", "format"],
    ]
    for cmd in cmds:
        res = subprocess.run(cmd, text=True, capture_output=True)
        if res.returncode != 0:
            print(f"{Colors.RED}✗ Compile failed: {' '.join(cmd)}{Colors.ENDC}")
            if res.stdout.strip(): print(res.stdout.strip())
            if res.stderr.strip(): print(res.stderr.strip())
            return False
    print(f"{Colors.GREEN}✓ Compilation successful{Colors.ENDC}")
    return True


def detect_binaries():
    # handle both with and without .exe
    start = next((p for p in [Path("start"), Path("start.exe")] if p.exists()), None)
    fmt   = next((p for p in [Path("format"), Path("format.exe")] if p.exists()), None)
    if not start:
        raise FileNotFoundError("start/start.exe not found (did compile.sh run?).")
    if not fmt:
        raise FileNotFoundError("format/format.exe not found (did compile.sh run?).")
    return start, fmt

def prepare_work_dir(tc_dir: Path, start: Path, fmt: Path):
    # detect files in tc dir
    bif  = _pick_file(tc_dir, ["*hailfinder*.bif", "*.bif"])
    dat  = _pick_file(tc_dir, ["records*.dat", "*.dat"])
    gold = _pick_file(tc_dir, ["gold*hailfinder*.bif", "gold*.bif"])
    if not (bif and dat and gold):
        missing = []
        if not bif:  missing.append("BIF")
        if not dat:  missing.append("DAT")
        if not gold: missing.append("GOLD")
        raise FileNotFoundError(f"Missing in {tc_dir.name}: {', '.join(missing)}")

    # make fresh work dir
    work = WORK_ROOT / tc_dir.name
    if work.exists(): shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    # copy / normalize names
    shutil.copy2(bif,  work / BIF_TARGET_NAME)
    shutil.copy2(dat,  work / DAT_TARGET_NAME)
    shutil.copy2(gold, work / GOLD_TARGET_NAME)

    # copy binaries (keep their names, we will call the copied names)
    shutil.copy2(start, work / start.name)
    shutil.copy2(fmt,   work / fmt.name)

    return work, (work / start.name), (work / fmt.name)

def run_binary(cmd, cwd: Path, live_dots=True, timeout=None):
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if live_dots:
        print(f"{Colors.DIM}Running", end="", flush=True)
        while proc.poll() is None:
            print(".", end="", flush=True)
            time.sleep(0.5)
        print(Colors.ENDC, end="")
    out, err = proc.communicate()
    return proc.returncode, out, err

_fmt_pass_re   = re.compile(r"Format check passed\.", re.I)
_fmt_fail_re   = re.compile(r"Format check failed", re.I)
_total_err_re  = re.compile(r"Total learning error:\s*([+-]?\d+(\.\d+)?([eE][+-]?\d+)?)")

def parse_format_output(text: str):
    passed = bool(_fmt_pass_re.search(text)) and not bool(_fmt_fail_re.search(text))
    m = _total_err_re.search(text)
    te = float(m.group(1)) if m else None
    return passed, te

def run_one_case(tc_dir: Path, results: list, start_bin: Path, fmt_bin: Path):
    name = tc_dir.name
    print_section(f"Running Test Case: {name}")
    info = {
        "name": name,
        "status": "FAILED",
        "time": 0.0,
        "format_passed": False,
        "total_error": None,
        "error": None,
    }

    try:
        work, start_path, fmt_path = prepare_work_dir(tc_dir, start_bin, fmt_bin)
    except Exception as e:
        info["error"] = str(e)
        print(f"{Colors.RED}✗ {e}{Colors.ENDC}")
        results.append(info)
        return

    t0 = time.time()

    # 1) run start (must produce solved.bif)
    print(f"{Colors.YELLOW}Training (start)…{Colors.ENDC}")
    rc, out1, err1 = run_binary(["./" + start_path.name, "hailfinder.bif", "records.dat"], cwd=work, live_dots=True)
    if rc != 0:
        info["error"] = f"start failed (exit {rc})"
        tail = (err1 or out1 or "")[-1000:]
        print(f"{Colors.RED}✗ start failed{Colors.ENDC}\n{tail}")
        results.append(info)
        return

    solved_path = work / SOLVED_FILE_NAME
    if not solved_path.exists():
        info["error"] = f"{SOLVED_FILE_NAME} not created by start"
        print(f"{Colors.RED}✗ {SOLVED_FILE_NAME} not found in {work}{Colors.ENDC}")
        results.append(info)
        return

    # 2) run format checker
    print(f"{Colors.YELLOW}Checking format/score (format)…{Colors.ENDC}")
    rc2, out2, err2 = run_binary([f"./{fmt_path.name}"], cwd=work, live_dots=False)
    combined = (out2 or "") + "\n" + (err2 or "")
    passed, te = parse_format_output(combined)

    info["format_passed"] = passed
    info["total_error"] = te
    if passed:
        info["status"] = "PASSED"

    info["time"] = time.time() - t0
    print(f"{Colors.GREEN if rc2==0 else Colors.RED}Format checker exit={rc2}{Colors.ENDC}")
    print(f"{Colors.CYAN}Format: {'PASSED' if passed else 'FAILED'}{Colors.ENDC}")
    if te is not None:
        print(f"{Colors.CYAN}Total learning error: {te:.6f}{Colors.ENDC}")

    # 3) stash solved.bif
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(solved_path, out_dir / SOLVED_FILE_NAME)

    results.append(info)
    # summary
    print(f"\n{Colors.BOLD}Result Summary:{Colors.ENDC}")
    print(f"  Status: {Colors.GREEN if info['status']=='PASSED' else Colors.RED}{info['status']}{Colors.ENDC}")
    print(f"  Time: {Colors.CYAN}{info['time']:.2f}s{Colors.ENDC}")
    print(f"  Format: {Colors.CYAN}{'PASSED' if info['format_passed'] else 'FAILED'}{Colors.ENDC}")
    if info['total_error'] is not None:
        print(f"  Total Error: {Colors.CYAN}{info['total_error']:.6f}{Colors.ENDC}")

def generate_report(results):
    print_section("Generating Report")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("result.log", "w") as f:
        f.write("="*70 + "\n")
        f.write("BAYESIAN NETWORK TEST RESULTS\n")
        f.write(f"Generated: {ts}\n")
        f.write("="*70 + "\n\n")

        total = len(results)
        passed = sum(1 for r in results if r["status"] == "PASSED")
        f.write(f"Summary: {passed}/{total} tests passed\n")
        f.write("-"*70 + "\n\n")

        for r in results:
            f.write(f"Test Case: {r['name']}\n")
            f.write(f"  Status: {r['status']}\n")
            f.write(f"  Execution Time: {r['time']:.2f}s\n")
            f.write(f"  Format Check: {'PASSED' if r['format_passed'] else 'FAILED'}\n")
            if r["total_error"] is not None:
                f.write(f"  Total Error: {r['total_error']:.6f}\n")
            if r["error"]:
                f.write(f"  Error: {r['error']}\n")
            f.write("\n" + "-"*70 + "\n\n")

        avg_time = sum(r["time"] for r in results) / total if total else 0.0
        f.write("Overall Statistics:\n")
        f.write(f"  Total Tests: {total}\n")
        f.write(f"  Passed: {passed}\n")
        f.write(f"  Failed: {total - passed}\n")
        f.write(f"  Average Time: {avg_time:.2f}s\n")

        tes = [r["total_error"] for r in results if r["total_error"] is not None]
        if tes:
            f.write(f"  Average Total Error: {sum(tes)/len(tes):.6f}\n")
            f.write(f"  Min Total Error: {min(tes):.6f}\n")
            f.write(f"  Max Total Error: {max(tes):.6f}\n")

    print(f"{Colors.GREEN}✓ Report generated: result.log{Colors.ENDC}")

    # console summary
    print(f"\n{Colors.BOLD}{Colors.CYAN}Final Summary:{Colors.ENDC}\n")
    print(f"{'Test Case':<18} {'Status':<10} {'Time':<9} {'Format':<8} {'TotalErr':<10}")
    print("-" * 60)
    for r in results:
        status = f"{Colors.GREEN}PASSED{Colors.ENDC}" if r["status"]=="PASSED" else f"{Colors.RED}FAILED{Colors.ENDC}"
        testr = f"{r['total_error']:.6f}" if r["total_error"] is not None else "N/A"
        print(f"{r['name']:<18} {status:<10} {r['time']:<9.2f} {('✓' if r['format_passed'] else '✗'):<8} {testr:<10}")
    print(f"\n{Colors.BOLD}Total: {passed}/{len(results)} passed{Colors.ENDC}")

def main():
    print_banner()

    if not Path("compile.sh").exists():
        print(f"{Colors.RED}Error: compile.sh not found in current directory.{Colors.ENDC}")
        return

    tcs = find_test_cases()
    if not tcs:
        print(f"{Colors.RED}No test cases found under {TC_ROOT}{Colors.ENDC}")
        return

    display_test_cases(tcs)
    selected = select_test_cases(tcs)
    print(f"\n{Colors.GREEN}Selected {len(selected)} test case(s){Colors.ENDC}")

    confirm = input(f"\n{Colors.YELLOW}Proceed with testing? (y/n): {Colors.ENDC}").strip().lower()
    if confirm != "y":
        print(f"{Colors.YELLOW}Testing cancelled.{Colors.ENDC}")
        return

    print_section("Compilation")
    if not run_compile():
        print(f"{Colors.RED}Compilation failed. Exiting.{Colors.ENDC}")
        return

    try:
        start_bin, fmt_bin = detect_binaries()
    except Exception as e:
        print(f"{Colors.RED}{e}{Colors.ENDC}")
        return

    results = []
    t0 = time.time()
    for i, tc in enumerate(selected, 1):
        print(f"\n{Colors.HEADER}{Colors.BOLD}[{i}/{len(selected)}]{Colors.ENDC}")
        run_one_case(tc, results, start_bin, fmt_bin)
    total_time = time.time() - t0

    generate_report(results)
    print(f"\n{Colors.BOLD}{Colors.GREEN}All tests completed in {total_time:.2f}s{Colors.ENDC}")
    print(f"{Colors.CYAN}See result.log and test_case_checker/output/* for artifacts.{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Testing interrupted by user.{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Unexpected error: {e}{Colors.ENDC}")
        import traceback; traceback.print_exc()
        sys.exit(1)
