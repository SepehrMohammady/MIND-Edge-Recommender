"""Model files of the paper: the versioned folder models/ and, on request, an archive of all checkpoints.

models/ holds the files that ran on the boards and the phone (seed 42), each with the byte table its caller
looks up before the network, and models/manifest.json with the size, SHA-256, inputs and outputs, source file
and checkpoint of every file:

  <encoder>_body_int8qdq.onnx   8-bit QDQ file (ONNX Runtime static quantisation; scripts/export_int8.py and
                                scripts/deploy_boards.py)
  <encoder>_body_fp32.onnx      the same network at full precision
      inputs  embedded_bytes (1, 64, 128) float32: byte_table[ids] transposed, ids = UTF-8 bytes of the
              title, first 128, id = byte + 1, 0 = padding (src/student.text_to_bytes)
              mask (1, 1, 128) float32: 1 where ids != 0
      output  news_embedding (1, 384)
  byte_table_fixed.f32          257 x 64 float32 little-endian, shared by the seven encoders of Table 3
  <encoder>_byte_table.f32      the own table of the two reference encoders and of the one-bit encoder
  user_encoder.onnx             additive attention of the app's checkpoint: history_vectors (1, H, 384) and
                                history_mask (1, H) in, user_vector (1, 384) out
  reactnet_64-5-384_seed42_lce.tflite
                                one-bit encoder for TensorFlow Lite Micro with the Larq Compute Engine ops
                                (scripts/export_binary.py, scripts/lce_convert.py): embedded bytes
                                (1, 1, 128, 64) and mask (1, 1, 128, 1) in, the 384-d vector out

    python -m scripts.export_release                  # models/
    python -m scripts.export_release --checkpoints    # also release/mind-edge-recommender-checkpoints.zip
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import onnx

from src.board_models import byte_table

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts/stedgeai/models"
APP = ROOT / "artifacts/app/edge_encoder_v1"
LCE = ROOT / "artifacts/binary_lce"
OUT = ROOT / "models"
REL = ROOT / "release"

# name in models/, source stem in artifacts/stedgeai/models, checkpoint, what it is, where the paper reports it
ENCODERS = [
    ("unas_h7_seed42", "unas_h7_seed42", "artifacts/runs/unas_full/mind_h7_distill_ft_en_seed42.pt",
     "muNAS 144, recipe choice under the STM32H7B3I-DK budget", "Table 3, Figs. 5 and 6a"),
    ("unas_h7_c140_seed42", "unas_h7_c140_seed42", "artifacts/runs/unas_full/mind_h7_c140_distill_ft_en_seed42.pt",
     "muNAS 140, final choice under the STM32H7B3I-DK budget", "Table 3, Figs. 5 and 6"),
    ("unas_f401_seed42", "unas_f401_seed42", "artifacts/runs/unas_full/mind_f401_distill_ft_en_seed42.pt",
     "muNAS 134, recipe choice under the NUCLEO-F401RE budget", "Table 3, Figs. 5 and 6a"),
    ("unas_f401_c142_seed42", "unas_f401_c142_seed42", "artifacts/runs/unas_full/mind_f401_c142_distill_ft_en_seed42.pt",
     "muNAS 142, final choice under the NUCLEO-F401RE budget", "Table 3, Figs. 5 and 6"),
    ("hand_64-2-384_seed42", "hand_64-2-384_seed42", "artifacts/runs/unas_full/hand_64-2-384_distill_ft_en_seed42.pt",
     "hand-designed 64-2-384, best within the STM32H7B3I-DK budget", "Table 3, Fig. 6a"),
    ("hand_32-5-384_seed42", "hand_32-5-384_seed42", "artifacts/runs/unas_full/hand_32-5-384_distill_ft_en_seed42.pt",
     "hand-designed 32-5-384, best within the NUCLEO-F401RE budget", "Table 3, Fig. 6a"),
    ("hand_64-5-384_seed42", "hand_64-5-384_seed42", "artifacts/runs/unas_full/hand_64-5-384_distill_ft_en_seed42.pt",
     "hand-designed 64-5-384 in the search's terms (fixed byte table, mask channel, global average pooling)",
     "Table 3, Fig. 6a"),
    ("ref_64-5-384_seed42", "micro_nas_64-5-384", "artifacts/runs/p1/distill_ft_en_seed42.pt",
     "reference byte-CNN 64-5-384, distilled start, English clicks", "Sect. 5, board results of the reference encoder"),
    ("ref_mixed_64-5-384_seed42", "ref_mixed_64-5-384_seed42", "artifacts/runs/p1/distill_ft_mixed_seed42.pt",
     "reference byte-CNN 64-5-384, distilled start, clicks shown in a random xMIND language; the phone app's encoder",
     "Sect. 5, phone results and replay"),
]
TABLE3 = {"unas_h7", "unas_h7_c140", "unas_f401", "unas_f401_c142", "hand_64-2-384", "hand_32-5-384", "hand_64-5-384"}


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def io_spec(path):
    m = onnx.load(str(path))

    def dims(t):
        return [d.dim_value or d.dim_param for d in t.type.tensor_type.shape.dim]
    return {"inputs": {i.name: dims(i) for i in m.graph.input}, "outputs": {o.name: dims(o) for o in m.graph.output},
            "opset": m.opset_import[0].version}


def put(files, name, src, **info):
    dst = OUT / name
    if isinstance(src, np.ndarray):
        dst.write_bytes(src.astype("<f4").tobytes())
        source = info.pop("source")
    else:
        shutil.copyfile(src, dst)
        source = rel(src)
    entry = {"bytes": dst.stat().st_size, "sha256": sha256(dst), "source": source, **info}
    if dst.suffix == ".onnx":
        entry.update(io_spec(dst))
    files[name] = entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", action="store_true", help="also zip the checkpoints of every run")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    files = {}
    for name, stem, ckpt, what, where in ENCODERS:
        for kind in ("int8qdq", "fp32"):
            src = SRC / f"{stem}_body_{kind}.onnx"
            put(files, f"{name}_body_{kind}.onnx", src, encoder=what, reported_in=where, checkpoint=ckpt,
                byte_table="byte_table_fixed.f32" if name.rsplit("_seed", 1)[0] in TABLE3 else f"{name}_byte_table.f32")
    tables = {name.rsplit("_seed", 1)[0]: byte_table(name.rsplit("_seed", 1)[0], 42)
              for name, *_ in ENCODERS if name.rsplit("_seed", 1)[0] in TABLE3}
    first = next(iter(tables.values()))
    if not all(np.array_equal(first, t) for t in tables.values()):
        raise SystemExit("the Table 3 encoders do not share one byte table")
    put(files, "byte_table_fixed.f32", first, source="news_encoder.embed.weight of the Table 3 checkpoints (seed 42)",
        shape=[257, 64])
    put(files, "ref_64-5-384_seed42_byte_table.f32", byte_table("ref_64-5-384", 42),
        source="news_encoder.embed.weight of artifacts/runs/p1/distill_ft_en_seed42.pt", shape=[257, 64])
    put(files, "ref_mixed_64-5-384_seed42_byte_table.f32", byte_table("ref_mixed_64-5-384", 42),
        source="news_encoder.embed.weight of artifacts/runs/p1/distill_ft_mixed_seed42.pt", shape=[257, 64])
    put(files, "user_encoder.onnx", APP / "user_encoder.onnx", checkpoint="artifacts/runs/p1/distill_ft_mixed_seed42.pt",
        encoder="user encoder (additive attention) of the phone app's checkpoint", reported_in="Sect. 5, replay and phone")
    params = np.load(LCE / "reactnet_64-5-384_seed42_params.npz")
    put(files, "reactnet_64-5-384_seed42_byte_table.f32", params["embed"],
        source="embed of artifacts/binary_lce/reactnet_64-5-384_seed42_params.npz", shape=list(params["embed"].shape))
    lce_info = json.loads((LCE / "reactnet_64-5-384_seed42_lce_convert.json").read_text(encoding="utf-8"))
    put(files, "reactnet_64-5-384_seed42_lce.tflite", LCE / "reactnet_64-5-384_seed42_lce.tflite",
        checkpoint="artifacts/runs/binary/reactnet_64-5-384_seed42.pt",
        encoder="one-bit 64-5-384 with ReActNet blocks, distilled start, English clicks",
        reported_in="Sect. 5, one-bit encoder on the STM32H7B3I-DK; Fig. 6b",
        inputs={"embedded_bytes": [1, 1, 128, 64], "mask": [1, 1, 128, 1]}, outputs={"news_embedding": [1, 384]},
        byte_table="reactnet_64-5-384_seed42_byte_table.f32", converter={k: lce_info[k] for k in ("tensorflow", "larq", "lce")})
    (OUT / "manifest.json").write_text(json.dumps({
        "byte_ids": "UTF-8 bytes of the title, first 128, id = byte + 1, 0 = padding (src/student.text_to_bytes)",
        "embedded_bytes": "byte_table[ids] as (1, 64, 128) for the ONNX files; mask = (ids != 0) as (1, 1, 128)",
        "files": files}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"models/: {len(files)} files, {sum(f['bytes'] for f in files.values()) / 1e6:.2f} MB")

    if a.checkpoints:
        REL.mkdir(exist_ok=True)
        zpath = REL / "mind-edge-recommender-checkpoints.zip"
        cks = sorted(p for d in ("p1", "matrix", "unas_full", "binary") for p in (ROOT / "artifacts/runs" / d).glob("*.pt"))
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in cks:
                z.write(p, rel(p))
            z.writestr("checkpoints_manifest.json", json.dumps(
                {rel(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in cks}, indent=1) + "\n")
        print(f"{rel(zpath)}: {len(cks)} checkpoints, {zpath.stat().st_size / 1e6:.1f} MB, sha256 {sha256(zpath)}")


if __name__ == "__main__":
    main()
