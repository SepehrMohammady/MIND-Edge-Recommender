"""Energy per title on a board from a USB power meter and the energy firmware's serial log (schedule step 5).

The meter sits between the laptop's USB port and the board, so it sees the whole board (ST-LINK, power LED,
regulators, MCU). The energy firmware (firmware/nucleo_f401re, app energy) alternates two phases with the
same board set-up: the encoder running back to back, and the MCU in STOP mode. The rest of the board draws
the same in both, so

  dP          = mean power in the loop phase - power in the STOP phase (at the USB input)
  dI          = dP / V_bus, the extra current drawn while the MCU runs the encoder (its STOP current, tens of
                microamperes, is neglected)
  E_usb       = dP * t_network, the energy per title drawn from the USB port (includes the regulators' loss)
  E_mcu       = 3.3 V * dI * t_network, the energy per title at the MCU's supply, only when the readings file
                marks the board's 3.3 V regulator as linear (the same current then flows on the MCU supply)

t_network is the network's run time from the DWT cycle counter (startup line of the serial log); a loop run
also copies the input (2.8 ms), which draws about the same power and is left out of the energy per title.
For the one-bit firmware (app bnn) the log gives the mean cycles of Invoke() over its timing runs instead, and
the byte-table lookup as the input preparation; its outputs are checked against LCE's interpreter.

    python -m scripts.board_energy paper/results/board/energy/NUCLEO-F401RE__unas_f401_c142_usbmeter_readings.json
Writes <readings stem without _usbmeter_readings>_energy.json next to it.
"""
import json
import re
import sys
import time
from pathlib import Path

from src import runlog
from src.config import load_config

V_MCU = 3.3          # the boards' regulated MCU supply (nominal; the ST-LINK read 3.25 V on the F401, 3.28 V on the H7)

path = Path(sys.argv[1])
r = json.loads(path.read_text(encoding="utf-8"))
log = path.with_name(path.name.replace("_usbmeter_readings.json", "_serial.log")).read_text(encoding="utf-8-sig")
F_CPU = int(re.search(r"(\d+) Hz", log).group(1))                       # CPU clock printed by the firmware
bench = re.search(r"bench: (\d+) runs, invoke mean (\d+)", log)
if bench:                                                                # one-bit firmware (main_bnn.cc)
    net_cycles = int(bench.group(2))
    looks = [int(x) for x in re.findall(r"title \d+ lookup (\d+) cycles", log)]
    copy_cycles = round(sum(looks) / len(looks))
    diffs = [int(x) * 1e-9 for x in re.findall(r"maxdiff_e9 (\d+)", log)]
    differing = sum(d > 1e-5 for d in diffs)
else:
    net_cycles = int(re.search(r"network (\d+) cycles", log).group(1))
    copy_cycles = int(re.search(r"input copy (\d+) cycles", log).group(1))
    differing = int(re.search(r"output check: (\d+) of 384 values differ", log).group(1))
loop = re.search(r"loop: (\d+) runs in (\d+) ms, mean (\d+) cycles per run", log)

loop_p = [x["p_bus"] for x in r["readings"] if x["phase"] == "loop"]
stop_p = [x["p_bus"] for x in r["readings"] if x["phase"] == "stop"]
v_bus = sum(x["v_bus"] for x in r["readings"]) / len(r["readings"])
dp = sum(loop_p) / len(loop_p) - sum(stop_p) / len(stop_p)
di = dp / v_bus
t_net = net_cycles / F_CPU
out = {
    "board": r["board"], "meter": r["meter"], "readings_file": path.name,
    "network_cycles": net_cycles, "network_ms": round(1000 * t_net, 3),
    "input_copy_cycles": copy_cycles, "input_copy_ms": round(1000 * copy_cycles / F_CPU, 3),
    "loop_phase": ({"runs": int(loop.group(1)), "ms": int(loop.group(2)), "mean_cycles_per_run": int(loop.group(3)),
                    "mean_ms_per_run": round(1000 * int(loop.group(3)) / F_CPU, 3)} if loop else None),
    "output_check_values_differing": differing,
    "p_loop_w": loop_p, "p_stop_w": stop_p, "v_bus": round(v_bus, 4),
    "delta_p_w": round(dp, 4), "delta_i_ma": round(1000 * di, 2),
    "energy_per_title_usb_mj": round(1000 * dp * t_net, 3),
    "regulator": r.get("regulator"), "f_cpu_hz": F_CPU,
    **({"p_mcu_w": round(V_MCU * di, 4), "energy_per_title_mcu_mj": round(1000 * V_MCU * di * t_net, 3), "v_mcu": V_MCU}
       if str(r.get("regulator", "")).split()[0:2] in (["linear"], ["linear:"], ["assumed", "linear"]) else {}),
}
dst = path.with_name(path.name.replace("_usbmeter_readings.json", "_energy.json"))
dst.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
cfg = load_config()
cfg["seed"] = 42
keys = [k for k in ("network_ms", "delta_p_w", "delta_i_ma", "energy_per_title_usb_mj", "p_mcu_w",
                    "energy_per_title_mcu_mj", "loop_phase") if k in out]
runlog.append(cfg, f"board/{dst.stem}", {"meter": r["meter"], "read_by": r["read_by"], "phase_times": r["phase_times"],
              "regulator": r.get("regulator"), "f_cpu_hz": F_CPU}, {k: out[k] for k in keys}, time.time())
print(json.dumps({k: out[k] for k in keys}, indent=1))
