import os
import subprocess
import sys

# ================= 配置区域 =================
# 项目结构配置
RTL_DIR = "rtl"           # Verilog 源码目录
TB_DIR = "tb"             # Testbench 目录
SIM_DIR = "sim"           # 仿真输出目录
TOP_MODULE = "vector_dot_product"  # 顶层模块名 (不带.v)
TB_MODULE = "tb_" + TOP_MODULE    # Testbench 模块名

# 文件路径
RTL_FILES = [
    os.path.join(RTL_DIR, f"{TOP_MODULE}.v")
    # 如果有其他依赖文件，继续加在这里，例如: os.path.join(RTL_DIR, "defines.vh")
]
TB_FILE = os.path.join(TB_DIR, f"{TB_MODULE}.v")
OUT_FILE = os.path.join(SIM_DIR, "sim.out")
WAVE_FILE = os.path.join(SIM_DIR, "wave.vcd")


# ================= 脚本逻辑 =================
def run_command(cmd):
    """运行系统命令并检查是否成功"""
    print(f"[Exec] {cmd}")
    # shell=True 允许在 Windows 上运行复杂的 shell 命令
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error: Command failed with code {result.returncode}")
        sys.exit(1)


def main():
    # 1. 检查并创建 sim 目录
    if not os.path.exists(SIM_DIR):
        os.makedirs(SIM_DIR)
        print(f"✅ Created directory: {SIM_DIR}")

    # 2. 编译 (Icarus Verilog)
    # 构造命令: iverilog -o sim/sim.out -y rtl -I rtl tb/tb_xxx.v rtl/xxx.v
    print("🚀 Compiling...")
    rtl_sources = " ".join(RTL_FILES)
    # -g2012 开启 SystemVerilog 支持(可选)，-y 指定库目录
    compile_cmd = f"iverilog -o {OUT_FILE} -y {RTL_DIR} -I {RTL_DIR} {TB_FILE} {rtl_sources}"
    run_command(compile_cmd)
    print("✅ Compilation Successful.")

    # 3. 运行仿真 (VVP)
    print("RUNNING SIMULATION...")
    # -n 表示仿真结束后自动 finish，不用手动退出的交互模式
    sim_cmd = f"vvp -n {OUT_FILE}"
    run_command(sim_cmd)
    print("✅ Simulation Finished.")

    # 4. 打开波形 (GTKWave)
    # 检查波形文件是否生成
    if os.path.exists(WAVE_FILE):
        print("🌊 Opening Waveform...")
        # 使用 start 在新窗口打开，不阻塞当前终端
        if sys.platform == "win32":
            os.system(f"start gtkwave {WAVE_FILE}")
        else:
            os.system(f"gtkwave {WAVE_FILE} &")
    else:
        print("⚠️ Warning: Waveform file not found. Did you use $dumpfile in your TB?")


if __name__ == "__main__":
    main()
