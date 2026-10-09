"""Write the result tables of the course from the paper's result files, so the course quotes the
same numbers as the paper:

  lesson 5  architecture x precision table: matrix rerun (paper/results/matrix_summary.json)
  lesson 6  the full matrix with the reduced NRMS of Table 2 (p1_summary.json), quiz question 1,
            and the 14-language chart: distilled start + mixed-language clicks, three seeds
            (p1_summary.json), with the scratch encoder's mean for comparison

    python -m scripts.sync_course
"""
import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "paper/results"
cell = {(r["arm"], r["precision"]): r for r in json.loads((RES / "matrix_summary.json").read_text(encoding="utf-8"))}
nrms = json.loads((RES / "p1_summary.json").read_text(encoding="utf-8"))["nrms/nrms_reduced"]
ARM = {"nas": "NAS", "micro_nas": "Micro-NAS", "binarized_micro_nas": "Bin.µNAS"}
PREC = {"fp32": "FP32", "int8": "INT8", "binary": "Binary"}


def f(x, nd=3):
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def energy(r):
    return f(r["energy_uj"], 1) if r["energy_uj"] >= 100 else f(r["energy_uj"], 2)


best = {a: max(cell[(a, p)]["auc"] for p in PREC) for a in ARM}
mi8 = cell[("micro_nas", "int8")]

# lesson 5: AUC, size, RAM estimate, energy; the deployable cells as before (the six of the June table)
rows5 = []
for a, p in (("nas", "fp32"), ("nas", "int8"), ("nas", "binary"), ("micro_nas", "int8"), ("micro_nas", "binary"),
             ("binarized_micro_nas", "binary")):
    r = cell[(a, p)]
    hl = (a, p) == ("micro_nas", "int8")
    auc = f"{f(r['auc'])} ± {f(r['auc_sd'])}"
    auc = f"<strong>{auc}</strong>" if r["auc"] == best[a] or hl else auc
    size = f"{f(r['size_kb'], 0)} KB"
    rows5.append((f'    <tr class="row-hl"><td>{ARM[a]} / {PREC[p]} ⭐</td>' if hl else f"    <tr><td>{ARM[a]} / {PREC[p]}</td>")
                 + f"<td>{auc}</td><td>{'<strong>' + size + '</strong>' if hl else size}</td>"
                 f"<td>{f(r['ram_kb'], 0)} KB</td><td>{'<strong>' + energy(r) + '</strong>' if hl else energy(r)}</td></tr>")

# lesson 6: the nine cells and the reduced NRMS
rows6 = []
for a in ARM:
    for p in PREC:
        r = cell[(a, p)]
        hl = (a, p) == ("micro_nas", "int8")
        vals = [f"{f(r['auc'])} ± {f(r['auc_sd'])}", f(r["mrr"]), f(r["ndcg@10"]), f(r["size_kb"], 0), energy(r)]
        if hl:
            vals = [f"<strong>{v}</strong>" for v in vals]
        elif r["auc"] == best[a]:
            vals[0] = f"<strong>{vals[0]}</strong>"
        rows6.append(('    <tr class="row-hl">' if hl else "    <tr>") + f"<td>{ARM[a]}</td><td>{PREC[p]}</td>"
                     + "".join(f"<td>{v}</td>" for v in vals) + "</tr>")
rows6.append('    <tr style="border-top:2px solid #ccc"><td colspan="2"><em>NRMS، نسخهٔ کاهش‌یافته (FP32، سه seed)</em></td>'
             f"<td>{f(nrms['en_auc']['mean'])} ± {f(nrms['en_auc']['std'])}</td><td>{f(nrms['en_mrr']['mean'])}</td>"
             f"<td>{f(nrms['en_ndcg10']['mean'])}</td><td>~27,000 (INT8: 6,800)</td><td>زیاد</td></tr>")


def replace_tbody(html, marker, rows):
    i = html.index(marker)
    a = html.index("<tbody>", i) + len("<tbody>")
    b = html.index("</tbody>", a)
    return html[:a] + "\n" + "\n".join(rows) + "\n  " + html[b:]


p5 = ROOT / "course/05-quantization/index.html"
s = p5.read_text(encoding="utf-8")
s = replace_tbody(s, "<th>معماری + Precision</th>", rows5)
p5.write_text(s, encoding="utf-8")

p6 = ROOT / "course/06-evaluation/index.html"
s = p6.read_text(encoding="utf-8")
s = replace_tbody(s, "<h2>۶.۳ نتایج کامل", rows6)
q_old = re.search(r'<span class="q-num">۱</span> AUC=[0-9.]+ برای Micro-NAS/INT8 در مقایسه با '
                  r'(NRMS baseline|NRMS کاهش‌یافته) \([0-9.]+\)', s)
s = s.replace(q_old.group(0), f'<span class="q-num">۱</span> AUC={f(mi8["auc"])} برای Micro-NAS/INT8 در مقایسه با '
                              f'NRMS کاهش‌یافته ({f(nrms["en_auc"]["mean"])})')
e_old = re.search(r'<div class="explanation">[0-9.]+ (نسبت به|در برابر) [0-9.]+.*?</div>', s)
s = s.replace(e_old.group(0), f'<div class="explanation">{f(mi8["auc"])} در برابر {f(nrms["en_auc"]["mean"])}: '
                              f'تفاوتی در حد نوسان seedها (انحراف معیار {f(mi8["auc_sd"])} و {f(nrms["en_auc"]["std"])})؛ '
                              'مهم این است که این دقت با فقط 204 KB به دست می‌آید، در برابر ~27 MB (FP32) یا 6.8 MB (INT8) '
                              'برای NRMS. این مدل از صفر و فقط با کلیک‌های انگلیسی آموزش دیده و ترجمه‌ها را نزدیک به تصادف '
                              'می‌خواند (بخش ۶.۴).</div>')

# 14-language chart: distilled start + mixed-language clicks (Table 2 of the paper, three seeds)
P = json.loads((RES / "p1_summary.json").read_text(encoding="utf-8"))
mixed, scratch = P["student/distill_ft_mixed"], P["student/scratch_en"]
NAMES = {"en": "انگلیسی", "hat": "هائیتی کریول", "jpn": "ژاپنی", "ind": "اندونزیایی", "grn": "گوارانی",
         "ron": "رومانیایی", "tur": "ترکی", "vie": "ویتنامی", "swh": "سواحیلی", "zho": "چینی", "som": "سومالیایی",
         "tha": "تایلندی", "tam": "تامیلی", "fin": "فنلاندی", "kat": "گرجی"}
bars = [("en", mixed["en_auc"]["mean"])] + sorted(((l, v["mean"]) for l, v in mixed["per_language"].items()),
                                                key=lambda t: -t[1])
top = max(v for _, v in bars)
chart = ['<div class="bar-chart">']
for lang, v in bars:
    colour = "blue" if v >= 0.58 else "orange" if v >= 0.55 else "red"
    chart.append(f'  <div class="bar-row"><div class="bar-label">{lang} ({NAMES[lang]})</div><div class="bar-track">'
                 f'<div class="bar-fill {colour}" data-width="{round(100 * v / top)}%">{f(v)}</div></div></div>')
chart.append("</div>")
pl = mixed["per_language"]
intro = ("<p>انکودر بایتی با شروع از وزن‌های تقطیرشده، آموزش‌دیده روی کلیک‌هایی که هرکدام به یک زبان تصادفی نشان "
         f"داده می‌شوند (جدول ۲ مقاله، میانگین سه seed): انگلیسی {f(mixed['en_auc']['mean'])} و میانگین ۱۴ ترجمه "
         f"{f(mixed['xlang_auc']['mean'])}:</p>")
callout = ('<div class="callout callout-info">\n  <div class="callout-title">💡 تفسیر</div>\n  '
           f"همین انکودر اگر از صفر و فقط با کلیک‌های انگلیسی آموزش ببیند ترجمه‌ها را به‌طور میانگین با AUC "
           f"{f(scratch['xlang_auc']['mean'])} می‌خواند، نزدیک به تصادف؛ هم‌ترازی زبان‌ها از تقطیر می‌آید. "
           f"ضعیف‌ترین زبان‌ها خط غیرلاتین دارند: گرجی {f(pl['kat']['mean'])}، تامیلی {f(pl['tam']['mean'])} و "
           f"تایلندی {f(pl['tha']['mean'])}؛ چینی و ژاپنی با وجود خط غیرلاتین به {f(pl['zho']['mean'])} و "
           f"{f(pl['jpn']['mean'])} می‌رسند.\n</div>")
a = s.index("<h2>۶.۴ ارزیابی ۱۴ زبانه</h2>") + len("<h2>۶.۴ ارزیابی ۱۴ زبانه</h2>")
b = s.index("<h2>۶.۵")
s = s[:a] + "\n" + intro + "\n\n" + "\n".join(chart) + "\n\n" + callout + "\n\n" + s[b:]
p6.write_text(s, encoding="utf-8")

# lesson 4: the three June architectures, at the precision of their arm
p4 = ROOT / "course/04-architecture/index.html"
s = p4.read_text(encoding="utf-8")
rows4 = []
for a, p, star, arch in (("nas", "int8", "", (256, 4, 384)), ("micro_nas", "int8", " ⭐", (64, 5, 384)),
                          ("binarized_micro_nas", "binary", "", (96, 2, 384))):
    r = cell[(a, p)]
    auc = f"{f(r['auc'])} ± {f(r['auc_sd'])}" + (" (Binary)" if p == "binary" else "")
    auc = auc if p == "binary" else f'<span class="badge-good">{auc}</span>'
    name = {"nas": "NAS", "micro_nas": "Micro-NAS", "binarized_micro_nas": "Bin. µNAS"}[a]
    rows4.append(f"    <tr><td>{name}{star}</td><td>{arch[0]}</td><td>{arch[1]}</td><td>{arch[2]}</td><td>{auc}</td>"
                 f"<td>{f(r['size_kb'], 0)} KB</td></tr>")
s = replace_tbody(s, "<h2>۴.۴ معماری‌های یافته‌شده</h2>", rows4)
x_old = re.search(r"out_dim=384 یافت که AUC [0-9.]+ در 204 KB می‌دهد[^<]*", s)
s = s.replace(x_old.group(0), f"out_dim=384 یافت که AUC {f(mi8['auc'])} در 204 KB می‌دهد (میانگین سه seed). "
                              f"مرز ۲ میلیون MAC در آن جست‌وجو درست شمرده نشد و این معماری "
                              f"{f(cell[('micro_nas', 'fp32')]['macs'] / 1e6, 2)} میلیون MAC دارد؛ "
                              "جست‌وجوی µNAS درس ۷ مرز را رعایت می‌کند.")
p4.write_text(s, encoding="utf-8")

# lesson 5: the binary callout, from the matrix rerun and the three-seed ReActNet runs (scripts/run_binary.py)
rn = json.loads((RES / "binary_reactnet.json").read_text(encoding="utf-8"))["summary"]
nb = cell[("micro_nas", "binary")]
text = (f"Binaryِ ساده هزینه دارد: Micro-NAS/Binary به <strong>{f(nb['auc'])} ± {f(nb['auc_sd'])}</strong> می‌رسد "
        f"(میانگین سه seed؛ اجرای تکیِ ژوئن 0.521 بود و «نزدیک به تصادف» به نظر می‌رسید). با تکنیک‌های "
        f"<strong>ReActNet</strong> (RSign/RPReLU) + میان‌بر <strong>Bi-Real</strong> و مقداردهیِ اولیه با distillation، "
        f"دقتِ binary به <strong>{f(rn['auc'])} ± {f(rn['auc_sd'])}</strong> در {f(rn['cost']['size_kb'], 0)}KB می‌رسد: "
        f"{f(rn['auc'] - nb['auc'])} بالاتر از binaryِ ساده و {f(nrms['en_auc']['mean'] - rn['auc'])} پایین‌تر از NRMSِ "
        "کاهش‌یافته (27 MB). Binary گزینهٔ کم‌ترین‌فوت‌پرینت است؛ INT8 هنوز در دقت جلوتر.")
p5 = ROOT / "course/05-quantization/index.html"
s = p5.read_text(encoding="utf-8")
i = s.index('<div class="callout-title">✅ بازیابی دقت Binary (هدف اصلی ما)</div>')
a = s.index("</div>", i) + len("</div>")
b = s.index("</div>", a)
s = s[:a] + "\n  " + text + "\n" + s[b:]
p5.write_text(s, encoding="utf-8")

# lesson 7: the µNAS encoders on both boards (paper/results/unas_summary.json, Table 3) and the phone timing
U = json.loads((RES / "unas_summary.json").read_text(encoding="utf-8"))
budget = {"h7": "بودجهٔ H7", "f401": "بودجهٔ F401", "none": "بیرون از هر دو بودجه"}


def label7(r):
    if r["name"].startswith("mind_"):
        kind = "انتخاب نهایی" if "final choice" in r["label"] else "انتخاب با دستور کوتاه"
        idx = re.search(r"(\d+)", r["label"])[1]
        return f"µNAS {idx} ({kind})، {budget[r['group']]}"
    return f"دست‌ساز {r['name'][len('hand_'):]}، {budget[r['group']]}"


rows7 = []
for r in U:
    b = r["boards"]
    rows7.append(f"    <tr><td>{label7(r)}</td><td>{r['macs'] / 1e6:.2f} M</td><td>{f(r['en_auc']['mean'])}</td>"
                 f"<td>{b['STM32H7B3I-DK']['duration_ms']:.1f} ms</td><td>{b['NUCLEO-F401RE']['duration_ms']:.1f} ms</td></tr>")
p7 = ROOT / "course/07-deployment/index.html"
s = p7.read_text(encoding="utf-8")
s = replace_tbody(s, "<h3>روش ۵: جست‌وجوی µNAS زیر بودجهٔ هر برد</h3>", rows7)
g = {r["name"]: r for r in U}
fin = {r["group"]: r for r in U if "final choice" in r["label"]}
ms = lambda r, b: r["boards"][b]["duration_ms"]                                   # noqa: E731
para = (f"<p>دستور کوتاه جست‌وجو (96 هزار سطر، حداکثر 15 دوره) نامزدها را با ترتیبی می‌چیند که پس از آموزش کامل پابرجا "
        f"نمی‌ماند؛ برای همین انتخاب نهایی میان همان هشت نامزد برتر، پس از تقطیر کامل انجام شد. انتخاب‌های نهایی "
        f"به AUC {f(fin['h7']['en_auc']['mean'])} و {f(fin['f401']['en_auc']['mean'])} می‌رسند، در برابر "
        f"{f(g['hand_64-2-384']['en_auc']['mean'])} و {f(g['hand_32-5-384']['en_auc']['mean'])} برای بهترین شکل دست‌سازِ "
        f"همان بودجه. روی NUCLEO-F401RE انتخاب نهایی بودجهٔ F401 در {ms(fin['f401'], 'NUCLEO-F401RE'):.1f} میلی‌ثانیه "
        f"اجرا می‌شود، {ms(g['hand_32-5-384'], 'NUCLEO-F401RE') / ms(fin['f401'], 'NUCLEO-F401RE'):.1f} برابر سریع‌تر "
        f"از 32-5-384؛ در آن شکل دست‌ساز لایه‌های depthwise روی Cortex-M4 حدود 22 چرخه برای هر MAC می‌گیرند.</p>")
a = s.index("</table>", s.index("<h3>روش ۵: جست‌وجوی µNAS زیر بودجهٔ هر برد</h3>"))
a = s.index("</div>", a) + len("</div>")
b = s.index("</p>", a) + len("</p>")
s = s[:a] + "\n" + para + s[b:]
bench = RES / "phone" / "edge_bench"
lat = {(r["kind"], r["threads"]): r for r in
       json.loads(sorted(bench.glob("latency_*.json"))[-1].read_text(encoding="utf-8"))["rows"]}
phone = (f"همان فایل .onnx با onnxruntime-android اجرا می‌شود؛ برنامهٔ FeedWell-Edge میزبان مدل است. روی DOOGEE S98Pro "
         f"(MediaTek Helio G96) فایل 8-bit برای هر عنوان {f(lat[('int8', 1)]['medianMs'], 2)} میلی‌ثانیه طول می‌کشد "
         f"(میانهٔ 2000 اجرا، یک هسته) و فایل FP32 {f(lat[('fp32', 1)]['medianMs'], 2)} میلی‌ثانیه؛ بردارها با لپ‌تاپ "
         f"یکی‌اند. مصرف باتری هنوز اندازه‌گیری نشده است.")
s = re.sub(r"همان فایل \.onnx با onnxruntime-android اجرا می‌شود؛[^<]*", phone, s, count=1)
p7.write_text(s, encoding="utf-8")
print("lesson 5 rows:", len(rows5), "lesson 6 rows:", len(rows6), "lesson 4 rows:", len(rows4), "lesson 7 rows:", len(rows7))
