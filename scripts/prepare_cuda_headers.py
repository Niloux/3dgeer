"""Prepare project-local CUDA 12.8 headers for glibc 2.41+ / 2.43+."""

import os
from pathlib import Path
import re
import shutil


cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.8"))
source = cuda_home / "include"
target = Path(__file__).resolve().parents[1] / ".cache/cuda-include"
shutil.copytree(source, target, dirs_exist_ok=True)
header = target / "crt/math_functions.h"
text = header.read_text()
glibc = tuple(map(int, os.confstr("CS_GNU_LIBC_VERSION").split()[1].split(".")))
functions = []
if glibc >= (2, 41):
    functions += ["sinpi", "sinpif", "cospi", "cospif"]
if glibc >= (2, 43):
    functions += ["rsqrt", "rsqrtf"]
for name in functions:
    text, count = re.subn(
        rf"^(extern __DEVICE_FUNCTIONS_DECL__[^\n]*\b{name}\([^\n;]*\));$",
        r"\1 noexcept(true);",
        text,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise RuntimeError(f"Expected one CUDA 12.8 declaration for {name}, got {count}")
header.write_text(text)
print(f"Prepared CUDA headers in {target} ({len(functions)} declarations patched)")
