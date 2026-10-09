"""Energy per title of the on-device encoder, from the battery test of the FeedWell-Edge research screen.

Input: the newest paper/results/phone/edge_bench/battery_*.json (pulled from the phone): 15 min with the
app open and the encoder idle, then 15 min with the encoder running over the test titles in a loop (8-bit
file, one thread), the phone unplugged, screen kept on, battery read every 30 s through Android's
BatteryManager and the battery broadcast.

This phone (DOOGEE S98Pro) reports no instantaneous current (CURRENT_NOW is always 0) and its charge counter
moves in steps of 29.46 mAh; the fuel gauge's average current (CURRENT_AVERAGE) changes from sample to
sample. Power per sample = |average current| x voltage; the first sample of each phase is skipped (it
covers the time before the phase). Energy per title = (mean power while encoding - mean power while idle)
/ titles per second. Two coarse checks are kept next to it: the capacity percentage (1 % steps; the charge
counter at full charge gives the capacity) and the charge counter itself.

    python -m scripts.phone_energy
Writes paper/results/phone_energy.json.
"""
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
src = sorted((ROOT / "paper/results/phone/edge_bench").glob("battery_*.json"))[-1]
d = json.loads(src.read_text(encoding="utf-8"))
out = {"source": str(src.relative_to(ROOT)), "file": d["kind"], "threads": d["threads"], "phases": {}}
for phase in ("idle", "load"):
    run = d[phase]
    s = run["samples"]
    if any(x["plugged"] != 0 for x in s):
        raise SystemExit(f"{phase}: the phone was plugged in during the phase")
    power = [abs(x["currentAvgUa"]) * x["voltageMv"] / 1e9 for x in s[1:]]
    seconds = (s[-1]["elapsedMs"] - s[0]["elapsedMs"]) / 1000
    out["phases"][phase] = {
        "seconds": round(seconds, 1), "titles": int(run["titles"]),
        "power_w": round(statistics.fmean(power), 4), "power_sd_w": round(statistics.pstdev(power), 4),
        "current_ma": round(statistics.fmean(abs(x["currentAvgUa"]) / 1000 for x in s[1:]), 1),
        "voltage_v": round(statistics.fmean(x["voltageMv"] / 1000 for x in s[1:]), 4),
        "capacity_pct_drop": s[0]["capacityPct"] - s[-1]["capacityPct"],
        "charge_counter_drop_mah": (s[0]["chargeCounterUah"] - s[-1]["chargeCounterUah"]) / 1000,
        "temperature_c": [s[0]["temperatureDeciC"] / 10, s[-1]["temperatureDeciC"] / 10]}
idle, load = out["phases"]["idle"], out["phases"]["load"]
rate = load["titles"] / load["seconds"]
extra_w = load["power_w"] - idle["power_w"]
full_mah = d["idle"]["samples"][0]["chargeCounterUah"] / 1000 / (d["idle"]["samples"][0]["capacityPct"] / 100)
by_pct = ((load["capacity_pct_drop"] - idle["capacity_pct_drop"]) / 100 * full_mah * 3.6
          * statistics.fmean([idle["voltage_v"], load["voltage_v"]]) / load["titles"])
by_counter = ((load["charge_counter_drop_mah"] - idle["charge_counter_drop_mah"]) * 3.6
              * statistics.fmean([idle["voltage_v"], load["voltage_v"]]) / load["titles"])
out.update({"titles_per_second": round(rate, 1), "ms_per_title": round(1000 / rate, 4),
            "extra_power_w": round(extra_w, 4), "energy_per_title_mj": round(extra_w / rate * 1000, 4),
            "check_capacity_pct_mj": round(by_pct * 1000, 4), "check_charge_counter_mj": round(by_counter * 1000, 4),
            "capacity_from_counter_mah": round(full_mah, 0)})
(ROOT / "paper/results/phone_energy.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
