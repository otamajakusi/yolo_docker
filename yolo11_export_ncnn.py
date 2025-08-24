import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# see ncnn/examples/yolo11.cpp


def pt_to_ncnn(model_path):
    model_dir = Path(model_path).parent
    model_stem = Path(model_path).stem

    # 2. export yolo11 torchscript
    subprocess.run(
        ["yolo", "export", f"model={model_path}", "format=torchscript"],
        check=True,
        cwd=model_dir,
    )
    # 3. convert torchscript with static shape
    subprocess.run(
        ["pnnx", f"{model_stem}.torchscript"],
        check=True,
        cwd=model_dir,
    )
    # 4. modify yolo11n_pnnx.py for dynamic shape inference
    model_pnnx_py_path = model_dir / f"{model_stem}_pnnx.py"
    pnnx_py = model_pnnx_py_path.read_text()
    pnnx_py = re.sub(
        r"(\s+v_235 = v_204\.view\(1, \d+), \d+\)",
        r"\g<1>, -1).transpose(1, 2)",
        pnnx_py,
        flags=re.MULTILINE,
    )
    pnnx_py = re.sub(
        r"(\s+v_236 = v_219\.view\(1, \d+), \d+\)",
        r"\g<1>, -1).transpose(1, 2)",
        pnnx_py,
        flags=re.MULTILINE,
    )
    pnnx_py = re.sub(
        r"(\s+v_237 = v_234\.view\(1, \d+), \d+\)",
        r"\g<1>, -1).transpose(1, 2)",
        pnnx_py,
        flags=re.MULTILINE,
    )
    pnnx_py = re.sub(
        r"(\s+)(v_238 = torch\.cat\(\(v_235, v_236, v_237\), dim=)2\)",
        r"\g<1>\g<2>1)\g<1>return v_238",
        pnnx_py,
        flags=re.MULTILINE,
    )
    pnnx_py = re.sub(
        r"(\s+v_96 = v_95\.view\(\d+, \d+, \d+, )\d+\)",
        r"\g<1>-1)",
        pnnx_py,
        flags=re.MULTILINE,
    )
    pnnx_py = re.sub(
        r"(\s+v_106 = v_105\.view\(\d+, \d+, )\d+, \d+\)",
        r"\g<1>v_95.size(2), v_95.size(3))",
        pnnx_py,
        flags=re.MULTILINE,
    )
    pnnx_py = re.sub(
        r"(\s+v_107 = v_99\.reshape\(\d+, \d+, )\d+, \d+\)",
        r"\g<1>v_95.size(2), v_95.size(3))",
        pnnx_py,
        flags=re.MULTILINE,
    )
    model_pnnx_py_path.write_text(pnnx_py)
    # 5. re-export yolo11 torchscript
    subprocess.run(
        [
            "python3",
            "-c",
            f"import {model_pnnx_py_path.stem}; {model_pnnx_py_path.stem}.export_torchscript()",
        ],
        check=True,
        cwd=model_dir,
    )

    # 6. convert new torchscript with dynamic shape
    subprocess.run(
        [
            "pnnx",
            f"{model_pnnx_py_path}.pt",
            "inputshape=[1,3,640,640]",
            "inputshape2=[1,3,320,320]",
        ],
        check=True,
        cwd=model_dir,
    )


def main(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        shutil.copy(model, tmpdir)
        pt_to_ncnn(Path(tmpdir) / Path(model).name)
        # 7. now you get ncnn model files
        model_dir = Path(model).parent
        model_stem = Path(model).stem
        ncnn_param = Path(tmpdir) / f"{model_stem}_pnnx.py.ncnn.param"
        ncnn_bin = Path(tmpdir) / f"{model_stem}_pnnx.py.ncnn.bin"
        shutil.copy(str(ncnn_param), str(model_dir))
        shutil.copy(str(ncnn_bin), str(model_dir))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="yolo11 pytorch model path")
    args = parser.parse_args()
    main(args.model)
