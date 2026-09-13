"""Shared builder for the single-variable Duck NVFP4 notebook variants.

Every candidate in `experiments/stage7_context_budget.md` is generated the same
way: take the pristine 10.69-baseline notebook, append one self-verifying block
to the customization-hook cell, and re-point the kernel id. Nothing else is
allowed to move, and this module enforces that rather than trusting the caller:

  * the baseline notebook's md5 is asserted before anything is read;
  * after patching, every cell except the hook cell is compared byte-for-byte
    against the baseline;
  * the hook cell must still contain its anchor line, so a notebook
    re-numbering can never silently patch the wrong cell;
  * the generated cell is compiled, so a broken f-string in a patch template is
    caught locally instead of 25 minutes into a GPU run.

That last guard is not hypothetical: the patch bodies are `.format()`
templates, so every literal brace in them has to be doubled, and a missed one
produces valid-looking source that fails at run time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build(*, base_nb: Path, out_nb: Path, meta: Path, baseline_md5: str,
          hook_cell: int, hook_anchor: str, patch: str, kernel_id: str) -> None:
    raw = base_nb.read_bytes()
    digest = hashlib.md5(raw).hexdigest()
    if digest != baseline_md5:
        raise SystemExit(
            f"{base_nb} md5 {digest} != baseline {baseline_md5}; the pristine "
            "baseline notebook has been modified"
        )

    base = json.loads(raw.decode("utf-8"))
    nb = json.loads(raw.decode("utf-8"))

    cell = nb["cells"][hook_cell]
    body = "".join(cell["source"])
    if hook_anchor not in body:
        raise SystemExit(f"cell {hook_cell} is not the customization hook")

    patched = body.rstrip("\n") + "\n" + patch
    compile(patched, f"<cell {hook_cell}>", "exec")  # catches template mistakes
    cell["source"] = patched.splitlines(keepends=True)

    for i, (x, y) in enumerate(zip(base["cells"], nb["cells"])):
        if i == hook_cell:
            continue
        if "".join(x["source"]) != "".join(y["source"]):
            raise SystemExit(f"cell {i} changed unexpectedly")
    if len(base["cells"]) != len(nb["cells"]):
        raise SystemExit("cell count changed")

    rendered = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    out_nb.parent.mkdir(parents=True, exist_ok=True)
    out_nb.write_text(rendered, encoding="utf-8")

    # `notebook/` is scratch -- whichever variant was generated last, and the
    # only thing `kaggle kernels push` looks at. Keep an addressable copy of
    # every variant so the repo records exactly what each kernel ran.
    slug = kernel_id.split("/")[-1]
    variant = out_nb.parent.parent / "variants" / f"{slug}.ipynb"
    variant.parent.mkdir(parents=True, exist_ok=True)
    variant.write_text(rendered, encoding="utf-8")
    print(f"variant archived -> {variant.relative_to(out_nb.parent.parent.parent)}")

    m = json.loads(meta.read_text(encoding="utf-8"))
    m["id"] = kernel_id
    m["title"] = kernel_id.split("/")[-1]
    meta.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"kernel id -> {kernel_id}")
