"""Unit tests for the Duck vLLM serving-configuration logic.

Covers the parts with real logic that could silently launch the WRONG server
configuration: argv flag removal (both spellings, duplicates), profile
application, the "unknown is not unsupported" rule that protects a GPU sweep
from a help-parsing failure, and the here-doc patch's anchor contract.

The baseline argv fixture below is the real production launch command, taken
from the bundled ``setup_commands.json`` of
``jakobbrggen/taaf-kaggle-source-anim-20260807-anim`` (the dataset our
submission notebook actually mounts), so these tests exercise the shape the
patch will really meet rather than a convenient invention.
"""

from __future__ import annotations

import json

import pytest

from kaggle_submission_duck.vllm_serving import (
    BASELINE_PROFILE,
    LAUNCH_ANCHOR,
    PROFILES,
    apply_mutation,
    build_setup_patch,
    drop_flag,
    patch_setup_command,
    profile_argv,
    speculative_config,
    unsupported_flags,
)

# The real production argv (see module docstring for provenance).
BASELINE_ARGV = [
    "/usr/bin/python", "-m", "vllm.entrypoints.openai.api_server",
    "--model", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1",
    "--served-model-name", "Qwen/Qwen3.8-27B-FP8",
    "--host", "127.0.0.1",
    "--port", "1234",
    "--tensor-parallel-size", "1",
    "--enable-auto-tool-choice",
    "--tool-call-parser", "qwen3_coder",
    "--generation-config", "vllm",
    "--enable-prefix-caching",
    "--default-chat-template-kwargs", '{"preserve_thinking": true}',
    "--reasoning-parser", "qwen3",
    "--max-model-len", "65536",
]


# --- drop_flag ---------------------------------------------------------------

def test_drop_valueless_flag_removes_only_that_flag():
    out = drop_flag(BASELINE_ARGV, "--enable-prefix-caching", has_value=False)
    assert "--enable-prefix-caching" not in out
    assert len(out) == len(BASELINE_ARGV) - 1
    # Nothing else moved or vanished.
    assert out == [t for t in BASELINE_ARGV if t != "--enable-prefix-caching"]


def test_drop_flag_with_value_removes_the_value_too():
    out = drop_flag(BASELINE_ARGV, "--max-model-len", has_value=True)
    assert "--max-model-len" not in out
    assert "65536" not in out
    assert len(out) == len(BASELINE_ARGV) - 2


def test_drop_flag_handles_equals_spelling():
    argv = ["a", "--kv-cache-dtype=fp8", "b"]
    assert drop_flag(argv, "--kv-cache-dtype", has_value=True) == ["a", "b"]


def test_drop_flag_removes_every_occurrence_not_just_the_first():
    # A duplicate left behind would silently re-enable what a profile turns off.
    argv = ["--enable-prefix-caching", "x", "--enable-prefix-caching", "y"]
    assert drop_flag(argv, "--enable-prefix-caching", has_value=False) == ["x", "y"]


def test_drop_flag_absent_is_a_noop():
    assert drop_flag(BASELINE_ARGV, "--not-present", has_value=False) == BASELINE_ARGV


def test_drop_flag_does_not_mutate_its_input():
    original = list(BASELINE_ARGV)
    drop_flag(BASELINE_ARGV, "--enable-prefix-caching", has_value=False)
    assert BASELINE_ARGV == original


def test_drop_flag_does_not_match_a_longer_flag_with_the_same_prefix():
    argv = ["--enable-prefix-caching-extra", "keep"]
    assert drop_flag(argv, "--enable-prefix-caching", has_value=False) == argv


# --- apply_mutation / profile_argv -------------------------------------------

def test_apply_mutation_drops_then_appends():
    out = apply_mutation(BASELINE_ARGV, drop=("--enable-prefix-caching",),
                         add=("--async-scheduling",))
    assert "--enable-prefix-caching" not in out
    assert out[-1] == "--async-scheduling"


def test_baseline_profile_is_an_exact_noop():
    assert profile_argv(BASELINE_ARGV, BASELINE_PROFILE) == BASELINE_ARGV


def test_mtp3_profile_appends_a_valid_speculative_config():
    out = profile_argv(BASELINE_ARGV, "mtp3")
    assert out[:len(BASELINE_ARGV)] == BASELINE_ARGV  # baseline preserved in order
    assert out[-2] == "--speculative-config"
    parsed = json.loads(out[-1])
    assert parsed == {"method": "mtp", "num_speculative_tokens": 3}


def test_flags_profile_turns_prefix_caching_off_exactly_once():
    out = profile_argv(BASELINE_ARGV, "flags")
    assert "--enable-prefix-caching" not in out
    assert out.count("--no-enable-prefix-caching") == 1
    assert "--async-scheduling" in out
    assert out[out.index("--kv-cache-dtype") + 1] == "fp8"


def test_combined_profile_adds_everything_its_parts_add():
    """The combined profile must be exactly its two parts, not a third thing.

    Compared on ADDED tokens and on the dropped flag separately -- a naive
    whole-argv set union is wrong here, because `mtp3` keeps
    `--enable-prefix-caching` while `flags` removes it.
    """
    baseline = set(BASELINE_ARGV)
    combined = profile_argv(BASELINE_ARGV, "mtp3+flags")
    added_combined = set(combined) - baseline
    added_parts = (
        (set(profile_argv(BASELINE_ARGV, "mtp3")) - baseline)
        | (set(profile_argv(BASELINE_ARGV, "flags")) - baseline)
    )
    assert added_combined == added_parts
    # And it inherits `flags`'s removal, which `mtp3` alone does not do.
    assert "--enable-prefix-caching" not in combined
    assert "--enable-prefix-caching" in profile_argv(BASELINE_ARGV, "mtp3")


def test_unknown_profile_raises():
    with pytest.raises(KeyError):
        profile_argv(BASELINE_ARGV, "does-not-exist")


@pytest.mark.parametrize("name", sorted(PROFILES))
def test_every_profile_produces_paired_flag_value_tokens(name):
    """No profile may leave a value-taking flag as the final token."""
    out = profile_argv(BASELINE_ARGV, name)
    for flag in ("--speculative-config", "--kv-cache-dtype"):
        if flag in out:
            assert out.index(flag) + 1 < len(out), f"{flag} has no value in {name}"


# --- speculative_config ------------------------------------------------------

def test_speculative_config_is_compact_single_token_json():
    value = speculative_config("mtp", 3)
    # Spaces would need shell quoting it never gets, since this is one argv token.
    assert " " not in value
    assert json.loads(value)["num_speculative_tokens"] == 3


def test_speculative_config_carries_extra_keys():
    value = speculative_config("ngram", 3, prompt_lookup_max=4, prompt_lookup_min=2)
    assert json.loads(value) == {
        "method": "ngram", "num_speculative_tokens": 3,
        "prompt_lookup_max": 4, "prompt_lookup_min": 2,
    }


@pytest.mark.parametrize("bad", [0, -1])
def test_speculative_config_rejects_nonpositive_token_counts(bad):
    with pytest.raises(ValueError):
        speculative_config("mtp", bad)


# --- unsupported_flags -------------------------------------------------------

def test_unsupported_flags_reports_missing_flags():
    missing = unsupported_flags(
        ["--speculative-config", "{}", "--async-scheduling"],
        {"--speculative-config"},
        help_usable=True,
    )
    assert missing == ["--async-scheduling"]


def test_unsupported_flags_ignores_non_flag_values():
    # The JSON value token must never be mistaken for an unsupported flag.
    assert unsupported_flags(["--speculative-config", '{"method":"mtp"}'],
                             {"--speculative-config"}, help_usable=True) == []


def test_unknown_is_not_unsupported_when_help_parsing_failed():
    """A help-parsing failure must not silently cancel an entire GPU sweep."""
    assert unsupported_flags(["--anything", "--at-all"], set(), help_usable=False) == []


# --- setup here-doc patch ----------------------------------------------------

FAKE_SETUP_COMMAND = (
    "def start_vllm_server() -> None:\n"
    "    cmd = [\n"
    "        '--max-model-len',\n"
    "        str(VLLM_MAX_MODEL_LEN),\n"
    "    ]\n"
    + LAUNCH_ANCHOR + "\n"
    "    process = subprocess.Popen(cmd, env=vllm_env())\n"
)


def test_baseline_patch_is_empty_and_leaves_the_command_untouched():
    assert build_setup_patch(BASELINE_PROFILE) == ""
    patched, count = patch_setup_command(FAKE_SETUP_COMMAND, BASELINE_PROFILE)
    assert count == 0
    assert patched == FAKE_SETUP_COMMAND


def test_patch_inserts_before_the_launch_anchor_exactly_once():
    patched, count = patch_setup_command(FAKE_SETUP_COMMAND, "mtp3")
    assert count == 1
    assert patched.count(LAUNCH_ANCHOR) == 1
    assert "cmd.extend(" in patched
    # The injected code must run BEFORE the server is launched, not after.
    assert patched.index("cmd.extend(") < patched.index(LAUNCH_ANCHOR)


def test_patch_reports_zero_when_the_anchor_is_missing():
    """A silently-unpatched command would launch the baseline while claiming otherwise."""
    patched, count = patch_setup_command("no anchor here", "mtp3")
    assert count == 0
    assert patched == "no anchor here"


def test_patched_command_is_valid_python():
    patched, _ = patch_setup_command(FAKE_SETUP_COMMAND, "mtp3+flags")
    compile(patched, "<patched-setup>", "exec")


def test_patched_command_actually_produces_the_expected_argv():
    """Execute the injected code against a real list to prove it does the job."""
    patch = build_setup_patch("flags")
    namespace = {"cmd": list(BASELINE_ARGV)}
    # The patch is written at function-body indentation; dedent to exec it.
    exec("\n".join(line[4:] for line in patch.splitlines()), namespace)
    assert namespace["cmd"] == profile_argv(BASELINE_ARGV, "flags")


def test_unknown_profile_patch_raises():
    with pytest.raises(KeyError):
        build_setup_patch("nope")
