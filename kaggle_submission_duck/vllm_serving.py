"""Pure logic for the Duck-harness vLLM serving-configuration change.

Extracted so it can be unit-tested (see ``tests/test_vllm_serving.py``)
independently of the Kaggle notebook it is inlined into -- the same
single-source-of-truth pattern already used by
``kaggle_submission_duck/duck_budget.py`` (Kaggle kernels have no import path
back into this repo, so the notebook must carry its own copy of the logic, and
that copy is generated from here rather than hand-typed twice).

Background (full method and numbers in ``experiments/stage7_duck_throughput.md``)
--------------------------------------------------------------------------------
A public fork (``keithtyser/duck-qwen3-8-flash-next-nvfp4-mtp``) states that
only model serving was changed versus the stock Duck harness, and reports a
large leaderboard gain. Its recipe is NVFP4 weights + native 3-token NEXTN MTP
speculative decoding + async scheduling + prefix caching off + a small pinned
KV cache + 8 vLLM sequences.

Two facts, both VERIFIED before any of this was written, determine what is
actually portable to our stack:

1. The fork's ``TAAF_VLLM_*`` environment variables are read **only** by its
   own source bundle's ``serving_setup.py``. Our bundle
   (``jakobbrggen/taaf-kaggle-source-anim-20260807-anim``) contains **zero**
   references to any of them, so setting those variables on our notebook is a
   silent no-op. Porting the recipe therefore means changing the **real vLLM
   argv**, which is what this module does.

2. Our **existing** production checkpoint already ships MTP weights: its
   ``config.json`` has ``text_config.mtp_num_hidden_layers = 1`` and the
   checkpoint contains ``mtp.safetensors`` (our own notebook already asserts
   that file is present). The NVFP4 model's MTP block has the same
   ``num_hidden_layers = 1``. So speculative decoding -- the recipe's single
   biggest lever -- is potentially reachable with **no model swap**, which is
   the hypothesis the benchmark kernel exists to test.

Our bundle launches vLLM from a hardcoded ``cmd = [...]`` list inside a
here-doc in ``setup_commands.json``. The notebook already rewrites that
here-doc's text (``_patch_qwen38_setup_commands`` swaps the model identity), so
appending serving flags is an extension of an existing, proven mechanism rather
than a new one.
"""

from __future__ import annotations

import json

# The anchor line in the bundle's setup here-doc, immediately after the vLLM
# ``cmd`` list is fully built and immediately before the server is spawned.
# Injecting here (rather than editing the list literal) means the patch does
# not depend on the list's internal formatting, only on this one line.
LAUNCH_ANCHOR = "    print('Starting vLLM OpenAI server:', ' '.join(cmd), flush=True)"

# Profile names are deliberately identical to the benchmark kernel's config
# names (``kaggle_submission_duck/notebook_serving_diag/serving_benchmark_cell.py``)
# so a row in the measured table maps 1:1 onto a value settable here.
BASELINE_PROFILE = "baseline"


def speculative_config(method: str, num_tokens: int, **extra: object) -> str:
    """Serialise a vLLM ``--speculative-config`` value.

    Compact separators matter: the JSON is passed as a single argv token, and
    a version with spaces would need shell-quoting it does not get.
    """
    if num_tokens < 1:
        raise ValueError(f"num_speculative_tokens must be >= 1, got {num_tokens}")
    payload: dict[str, object] = {"method": method, "num_speculative_tokens": num_tokens}
    payload.update(extra)
    return json.dumps(payload, separators=(",", ":"))


#: Each profile is ``(flags_to_remove, flags_to_append)``.
#: ``baseline`` is the empty no-op and is what ships until a measurement says
#: otherwise -- an unmeasured profile must never be the default.
PROFILES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    BASELINE_PROFILE: ((), ()),
    "mtp3": ((), ("--speculative-config", speculative_config("mtp", 3))),
    "mtp2": ((), ("--speculative-config", speculative_config("mtp", 2))),
    # Our checkpoint's MTP head is ONE layer deep. The fork drives 3 tokens
    # from an equally shallow head via its own runtime's
    # `index_share_for_mtp_iteration` (i.e. re-running the single layer), which
    # stock vLLM may not do -- in which case only 1 token is reachable here.
    # Kept as an explicit profile so that case is measurable rather than a
    # dead end.
    "mtp1": ((), ("--speculative-config", speculative_config("mtp", 1))),
    "qwen3_next_mtp1": (
        (),
        ("--speculative-config", speculative_config("qwen3_next_mtp", 1)),
    ),
    "flags": (
        ("--enable-prefix-caching",),
        ("--async-scheduling", "--no-enable-prefix-caching", "--kv-cache-dtype", "fp8"),
    ),
    # MEASURED WINNER (run 2, concurrency 37, real RTX PRO 6000):
    # 352.0 tok/s e2e vs. baseline 302.1 = +16.5%, the best of every
    # configuration tested. Note this is SUPERADDITIVE: mtp1 alone is +2.3%
    # and flags alone is +6.9% (sum +9.2%), so the combination is worth more
    # than its parts -- plausibly because --async-scheduling overlaps the
    # draft/verify work that speculative decoding adds. [INFERRED]
    #
    # Note the speculative depth: ONE token, not the public fork's three.
    # Our MTP head is one layer deep; driving it 3x drops the acceptance rate
    # from 0.719 to 0.456 and made throughput WORSE than baseline (-6.6%).
    "mtp1+flags": (
        ("--enable-prefix-caching",),
        (
            "--speculative-config", speculative_config("mtp", 1),
            "--async-scheduling", "--no-enable-prefix-caching",
            "--kv-cache-dtype", "fp8",
        ),
    ),
    "mtp3+flags": (
        ("--enable-prefix-caching",),
        (
            "--speculative-config", speculative_config("mtp", 3),
            "--async-scheduling", "--no-enable-prefix-caching",
            "--kv-cache-dtype", "fp8",
        ),
    ),
    "ngram3": (
        (),
        (
            "--speculative-config",
            speculative_config("ngram", 3, prompt_lookup_max=4, prompt_lookup_min=2),
        ),
    ),
    # The public fork's ENTIRE published profile ("kv5-bf16-mtp3-c8-cg32"),
    # transplanted onto our model. Included because the recipe is probably not
    # separable: speculative decoding pays off when decode is memory-bandwidth
    # bound (small batch) and costs extra compute when it is not, and the fork
    # pairs 3-token MTP with only EIGHT concurrent sequences and a 5 GiB KV
    # cache. Taking MTP without the small batch -- which is what `mtp3` alone
    # does -- may be testing an operating point the recipe never intended.
    "fork-profile": (
        ("--enable-prefix-caching",),
        (
            "--speculative-config", speculative_config("mtp", 3),
            "--async-scheduling",
            "--no-enable-prefix-caching",
            "--max-num-seqs", "8",
            "--max-num-batched-tokens", "8192",
            "--kv-cache-memory-bytes", "5368709120",
            "--max-cudagraph-capture-size", "32",
        ),
    ),
}


def drop_flag(argv: list[str], flag: str, has_value: bool) -> list[str]:
    """Remove ``flag`` (and its value, if any) from an argv list.

    Handles both ``--flag value`` and ``--flag=value`` spellings. Removing
    every occurrence rather than the first is deliberate: a duplicated flag
    left behind would silently re-enable the very thing a profile turns off.
    """
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            index += 2 if has_value else 1
            continue
        if token.startswith(flag + "="):
            index += 1
            continue
        out.append(token)
        index += 1
    return out


def apply_mutation(
    argv: list[str],
    *,
    drop: tuple[str, ...] | list[str] = (),
    add: tuple[str, ...] | list[str] = (),
) -> list[str]:
    """Return ``argv`` with ``drop`` flags removed and ``add`` tokens appended.

    Dropped flags are treated as valueless switches (``--enable-prefix-caching``
    and friends); a profile that needs to drop a flag *with* a value should
    call :func:`drop_flag` directly with ``has_value=True``.
    """
    out = list(argv)
    for flag in drop:
        out = drop_flag(out, flag, has_value=False)
    out.extend(add)
    return out


def profile_argv(argv: list[str], profile: str) -> list[str]:
    """Apply a named serving profile to a vLLM argv list."""
    if profile not in PROFILES:
        raise KeyError(f"unknown serving profile {profile!r}; known: {sorted(PROFILES)}")
    drop, add = PROFILES[profile]
    return apply_mutation(argv, drop=drop, add=add)


def unsupported_flags(
    add: list[str] | tuple[str, ...],
    supported: set[str],
    *,
    help_usable: bool,
) -> list[str]:
    """Flags in ``add`` that this vLLM build does not advertise.

    ``help_usable=False`` means the ``vllm serve --help`` dump could not be
    parsed. That is "unknown", not "unsupported", and must return ``[]`` --
    reporting unknown as unsupported would make one parsing failure silently
    cancel an entire GPU benchmark sweep.
    """
    if not help_usable:
        return []
    return [tok for tok in add if tok.startswith("--") and tok not in supported]


def build_setup_patch(profile: str) -> str:
    """Python source injected into the bundle's setup here-doc for ``profile``.

    Returns the empty string for the baseline profile so the patch is a literal
    no-op there. The emitted code logs the final argv, because a scored
    competition rerun produces no retrievable log and the printed line is the
    only record of what was actually launched.
    """
    if profile not in PROFILES:
        raise KeyError(f"unknown serving profile {profile!r}; known: {sorted(PROFILES)}")
    if profile == BASELINE_PROFILE:
        return ""
    drop, add = PROFILES[profile]
    # The profile name goes into its own variable rather than being
    # interpolated into a string literal: profile names and flag values both
    # contain quotes, and nesting them inside a generated literal produced a
    # SyntaxError in the emitted code (caught by test_patched_command_is_valid_python).
    lines = [
        "    # stage7-duck-throughput: serving profile (see",
        "    # experiments/stage7_duck_throughput.md for the measurements)",
        f"    _duck_serving_profile = {profile!r}",
        f"    for _duck_drop_flag in {list(drop)!r}:",
        "        while _duck_drop_flag in cmd:",
        "            cmd.remove(_duck_drop_flag)",
        f"    cmd.extend({list(add)!r})",
        "    print('taaf.kaggle: serving profile=', _duck_serving_profile,",
        "          'argv=', ' '.join(cmd), flush=True)",
    ]
    return "\n".join(lines)


def patch_setup_command(command: str, profile: str) -> tuple[str, int]:
    """Insert the serving-profile patch into one setup command's text.

    Returns ``(patched_command, n_substitutions)``. The caller is expected to
    raise when the count is not what it expects, matching the existing
    ``_patch_qwen38_setup_commands`` contract -- a silently-unpatched command
    would launch the baseline server while the notebook claimed otherwise.
    """
    patch = build_setup_patch(profile)
    if not patch:
        return command, 0
    if LAUNCH_ANCHOR not in command:
        return command, 0
    return command.replace(LAUNCH_ANCHOR, patch + "\n" + LAUNCH_ANCHOR, 1), 1
