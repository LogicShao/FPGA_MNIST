import argparse
import glob
import os
import subprocess
import shutil
import sys

# ================= Configuration =================
# Project layout
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RTL_DIR = os.path.join(BASE_DIR, "rtl")  # Verilog sources
TB_DIR = os.path.join(BASE_DIR, "tb")    # Testbenches
SIM_DIR = os.path.join(BASE_DIR, "sim")  # Simulation output
TOP_MODULE = "mnist_system_top"          # Default DUT module name (no .v)
TB_MODULE = "tb_" + TOP_MODULE           # Default TB module name

# File paths (resolved at runtime)
OUT_FILE = os.path.join(SIM_DIR, "sim.out")
WAVE_FILE = os.path.join(SIM_DIR, "wave.vcd")


def run_command(cmd, cwd=None):
    """Run a command and fail fast on error."""
    print(f"[Exec] {cmd}")
    # shell=True allows complex Windows shell commands
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"Error: Command failed with code {result.returncode}")
        sys.exit(1)


def collect_rtl_dirs(root_dir):
    rtl_dirs = [root_dir]
    for current, dirnames, _ in os.walk(root_dir):
        for dname in dirnames:
            rtl_dirs.append(os.path.join(current, dname))
    return rtl_dirs


def collect_extra_sources(root_dir):
    weights_dir = os.path.join(root_dir, "weights")
    rom_sources = glob.glob(os.path.join(weights_dir, "*_rom.v"))
    rtl_sources = glob.glob(os.path.join(root_dir, "*.v")) + glob.glob(os.path.join(root_dir, "*.sv"))
    return [os.path.abspath(p) for p in (rom_sources + rtl_sources)]


def resolve_tb_path(tb_arg):
    if tb_arg.endswith(".v") or os.path.sep in tb_arg or "/" in tb_arg:
        tb_path = tb_arg
    else:
        tb_path = os.path.join(TB_DIR, f"{tb_arg}.v")
    tb_path = os.path.abspath(tb_path)
    tb_module = os.path.splitext(os.path.basename(tb_path))[0]
    return tb_path, tb_module


def main():
    parser = argparse.ArgumentParser(description="Icarus Verilog simulation helper")
    parser.add_argument(
        "--tb",
        default=TB_MODULE,
        help="TB module name (without .v) or a TB file path",
    )
    parser.add_argument("--no-wave", action="store_true", help="Skip opening GTKWave")
    parser.add_argument("--fast", action="store_true", help="Enable FAST_SIM shortcuts")
    parser.add_argument("--quiet", action="store_true", help="Reduce TB prints and disable wave dump")
    args = parser.parse_args()

    tb_path, tb_module = resolve_tb_path(args.tb)
    if not os.path.exists(tb_path):
        print(f"Error: TB file not found: {tb_path}")
        sys.exit(1)

    # 1. Ensure sim directory exists
    if not os.path.exists(SIM_DIR):
        os.makedirs(SIM_DIR)
        print(f"Created directory: {SIM_DIR}")

    # 2. Compile (Icarus Verilog)
    print("Compiling...")
    rtl_dirs = collect_rtl_dirs(RTL_DIR)
    inc_flags = " ".join(f'-I "{d}"' for d in rtl_dirs)
    lib_flags = " ".join(f'-y "{d}"' for d in rtl_dirs)
    extra_sources = " ".join(f'"{p}"' for p in collect_extra_sources(RTL_DIR))
    fast_flag = "-DFAST_SIM" if args.fast else ""
    quiet_flag = "-DQUIET_SIM" if args.quiet else ""
    out_file = os.path.join(SIM_DIR, f"{tb_module}.out")
    compile_cmd = (
        f'iverilog -g2012 {fast_flag} {quiet_flag} -o "{out_file}" {inc_flags} {lib_flags} '
        f'{extra_sources} "{tb_path}"'
    )
    run_command(compile_cmd, cwd=BASE_DIR)
    print("Compilation successful.")

    # 3. Run simulation (VVP)
    print("Running simulation...")
    sim_cmd = f'vvp -n "{out_file}"'
    run_command(sim_cmd, cwd=BASE_DIR)
    print("Simulation finished.")

    # 4. Convert VCD to FST when possible (smaller and faster to open)
    if args.no_wave:
        return

    vcd_candidates = [
        os.path.join(SIM_DIR, f"{tb_module}.vcd"),
        os.path.join(BASE_DIR, f"{tb_module}.vcd"),
        WAVE_FILE,
    ]
    vcd_path = next((p for p in vcd_candidates if os.path.exists(p)), None)
    fst_path = None
    vcd2fst = shutil.which("vcd2fst")
    if vcd_path and vcd2fst:
        fst_path = os.path.splitext(vcd_path)[0] + ".fst"
        if not os.path.exists(fst_path):
            run_command(f'"{vcd2fst}" -o "{fst_path}" "{vcd_path}"', cwd=BASE_DIR)

    wave_candidates = [
        fst_path,
        os.path.join(SIM_DIR, f"{tb_module}.fst"),
        os.path.join(BASE_DIR, f"{tb_module}.fst"),
        vcd_path,
    ]
    wave_path = next((p for p in wave_candidates if p and os.path.exists(p)), None)
    if wave_path:
        print(f"Opening waveform: {wave_path}")
        if sys.platform == "win32":
            os.system(f'start "" gtkwave "{wave_path}"')
        else:
            os.system(f'gtkwave "{wave_path}" &')
    else:
        print("Warning: Waveform file not found. Did you use $dumpfile in your TB?")


if __name__ == "__main__":
    main()
