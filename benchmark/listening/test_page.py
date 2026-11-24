"""The local listening test page: one self-contained HTML file for one listener.

The page holds the consent sentence, the instructions, and one screen per trial. Each trial plays
two clips by relative path, asks the question the protocol names, and takes one of the protocol's
answers and an optional comment. At the end the listener saves a JSON answers file from the page.

The page text holds no external URL, no fetch call and no XMLHttpRequest, and the tests check
those strings. It has no analytics code. It names no condition: a clip is identified only by
its blinded file name, and the
mapping to a recording and a condition is in the sealed key, which the page generator reads and
the page does not contain.

The page generator is an operator step. It reads the sealed key, after checking the key's digest
against the build manifest, and writes the page into a directory beside the build directory.
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.atomic_write import write_atomically
from benchmark.listening.build import BUILD_NAME, CLIPS_DIR, KEY_NAME, SEALED_DIR
from benchmark.listening.protocol import ListeningProtocol
from benchmark.listening.trials import Trial, listener_trials

__all__ = [
    "ANSWERS_FORMAT",
    "PageError",
    "load_sealed_key",
    "page_html",
    "recordings_of",
    "write_test_page",
]

logger = logging.getLogger(__name__)

ANSWERS_FORMAT = "listening-answers-v1"
_PAGE_PREFIX = "listener_"
_PAGE_SUFFIX = ".html"


class PageError(RuntimeError):
    """Raised when a page cannot be generated: a key mismatch, a missing clip or a page."""


def load_sealed_key(build_dir: Path) -> dict[str, Any]:
    """The sealed key, after its digest is checked against the public build manifest.

    Raises:
        PageError: If the manifest or the key is unreadable, or the key's digest differs.
    """
    try:
        manifest: dict[str, Any] = json.loads((build_dir / BUILD_NAME).read_text())
        key_bytes = (build_dir / SEALED_DIR / KEY_NAME).read_bytes()
    except (OSError, json.JSONDecodeError) as error:
        logger.error("Cannot read the build in %s: %s", build_dir, error)
        raise PageError(f"cannot read the build: {error}") from error
    if hashlib.sha256(key_bytes).hexdigest() != manifest.get("key_sha256"):
        logger.error("The sealed key in %s does not match the build manifest", build_dir)
        raise PageError("the sealed key does not match the build manifest's digest")
    key: dict[str, Any] = json.loads(key_bytes)
    return key


def recordings_of(key: Mapping[str, Any]) -> tuple[str, ...]:
    """The recording identifiers in the key, sorted. The trial design uses this order."""
    return tuple(sorted({clip["identifier"] for clip in key["clips"]}))


def _blinded_ids(key: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    """The blinded identifier of each (recording, condition) pair.

    Raises:
        PageError: If a pair appears twice in the key.
    """
    mapping: dict[tuple[str, str], str] = {}
    for clip in key["clips"]:
        pair = (clip["identifier"], clip["condition"])
        if pair in mapping:
            raise PageError(f"the key maps {pair} twice")
        mapping[pair] = clip["blinded_id"]
    return mapping


def _clip_reference(build_dir: Path, pages_dir: Path, blinded_id: str) -> str:
    """The clip's path relative to the page, with forward slashes."""
    target = build_dir / CLIPS_DIR / f"{blinded_id}.wav"
    if not target.is_file():
        logger.error("Clip %s is missing from %s", blinded_id, build_dir)
        raise PageError(f"clip {blinded_id} is missing")
    return Path(os.path.relpath(target, pages_dir)).as_posix()


def _trial_payload(
    trials: Sequence[Trial],
    blinded: Mapping[tuple[str, str], str],
    references: Mapping[str, str],
) -> list[dict[str, str]]:
    """The page's trial list: an identifier and two clip paths per trial, nothing else."""
    payload = []
    for trial in trials:
        side_a = references[blinded[(trial.recording, trial.side_a)]]
        side_b = references[blinded[(trial.recording, trial.side_b)]]
        payload.append({"id": trial.trial_id, "a": side_a, "b": side_b})
    return payload


def _json_for_script(value: object) -> str:
    """JSON that is safe inside a script element: no markup can close the element early."""
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e")


def page_html(
    protocol: ListeningProtocol,
    listener_id: str,
    listener_index: int,
    trials: Sequence[Mapping[str, str]],
) -> str:
    """The page's HTML, with the protocol's wording and the listener's trial list.

    Args:
        protocol: The loaded protocol, which gives the consent sentence, the question and answers.
        listener_id: The pseudonymous ID the study assigned. No name or contact is accepted.
        listener_index: The listener's index in order of consent. The analysis uses it.
        trials: One mapping per trial with `id`, `a` and `b` (clip paths relative to the page).

    Returns:
        A complete HTML document with inline style and script.
    """
    config = {
        "format": ANSWERS_FORMAT,
        "protocolSha256": protocol.sha256,
        "listenerId": listener_id,
        "listenerIndex": listener_index,
        "question": protocol.trials.question,
        "answers": list(protocol.trials.answers),
        "minimumAge": protocol.listeners.minimum_age_years,
        "hearing": protocol.listeners.self_reported_hearing,
        "consent": protocol.listeners.consent_text,
        "trials": list(trials),
    }
    replacements = {
        "__CONSENT__": html.escape(protocol.listeners.consent_text),
        "__PLAYBACK__": html.escape(protocol.listeners.playback),
        "__QUESTION__": html.escape(protocol.trials.question),
        "__LISTENER__": html.escape(listener_id),
        "__CONFIG__": _json_for_script(config),
    }
    document = _TEMPLATE
    for placeholder, value in replacements.items():
        document = document.replace(placeholder, value)
    return document


def write_test_page(
    build_dir: Path,
    pages_dir: Path,
    protocol: ListeningProtocol,
    listener_index: int,
    listener_id: str,
) -> Path:
    """Write one listener's page into *pages_dir*, refusing to overwrite an earlier page.

    Args:
        build_dir: The published build directory.
        pages_dir: The directory the page goes into. Clip paths are written relative to it.
        protocol: The loaded protocol.
        listener_index: The listener's index in order of consent.
        listener_id: The listener's pseudonymous ID.

    Returns:
        The page's path.

    Raises:
        PageError: If the page exists already, the key does not match, or a clip is missing.
    """
    target = pages_dir / f"{_PAGE_PREFIX}{listener_index:03d}{_PAGE_SUFFIX}"
    if target.exists():
        raise PageError(f"page {target} already exists")
    key = load_sealed_key(build_dir)
    recordings = recordings_of(key)
    blinded = _blinded_ids(key)
    trials = listener_trials(protocol, listener_index, recordings)
    referenced = {
        blinded_id: _clip_reference(build_dir, pages_dir, blinded_id)
        for blinded_id in set(blinded.values())
    }
    payload = _trial_payload(trials, blinded, referenced)
    pages_dir.mkdir(parents=True, exist_ok=True)
    write_atomically(
        target, page_html(protocol, listener_id, listener_index, payload).encode("utf-8")
    )
    logger.info("Wrote the test page for listener %d to %s", listener_index, target)
    return target


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Listening test</title>
<style>
:root { --fg: #1d1d1f; --bg: #ffffff; --accent: #2a5db0; --muted: #6b6b70; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --fg: #f2f2f4; --bg: #161618; --accent: #7aa7f0; --muted: #a0a0a8;
  }
}
:root[data-theme="dark"] { --fg: #f2f2f4; --bg: #161618; --accent: #7aa7f0; --muted: #a0a0a8; }
body {
  margin: 0; padding: 16px; font-family: system-ui, sans-serif; color: var(--fg);
  background: var(--bg); line-height: 1.5;
}
main { max-width: 640px; margin: 0 auto; }
h1 { font-size: 1.4rem; }
button {
  font: inherit; padding: 8px 14px; margin: 4px 4px 4px 0; border-radius: 6px;
  border: 1px solid var(--accent); background: transparent; color: var(--fg); cursor: pointer;
}
button[disabled] { opacity: 0.4; cursor: default; }
button.chosen { background: var(--accent); color: var(--bg); }
.muted { color: var(--muted); }
audio { width: 100%; margin: 6px 0; }
textarea {
  width: 100%; min-height: 64px; font: inherit; box-sizing: border-box; color: var(--fg);
  background: var(--bg); border: 1px solid var(--muted); border-radius: 6px; padding: 6px;
}
.panel { border-top: 1px solid var(--muted); padding-top: 12px; margin-top: 12px; }
</style>
</head>
<body>
<main>
<h1>Listening test</h1>
<p class="muted">Listener __LISTENER__</p>

<section id="consent">
<h2>Before you start</h2>
<p id="consent-text">__CONSENT__</p>
<label><input type="checkbox" id="agree"> I agree to the statement above.</label><br>
<label><input type="checkbox" id="age"> I am at least
<span id="min-age"></span> years old.</label><br>
<label><input type="checkbox" id="hearing"> My hearing is normal.</label>
<p>Use __PLAYBACK__ for every trial. You can replay any clip as often as you like.</p>
<button id="start" disabled>Start</button>
</section>

<section id="trials" hidden>
<p class="muted"><span id="progress"></span></p>
<div class="panel">
<p>Play clip A and clip B, then answer:</p>
<p><strong id="question">__QUESTION__</strong></p>
<p><span>Clip A</span> <audio id="clip-a" controls preload="auto" aria-label="Clip A"></audio></p>
<p><span>Clip B</span> <audio id="clip-b" controls preload="auto" aria-label="Clip B"></audio></p>
<div id="choices"></div>
<label for="comment">Comment (optional)</label>
<textarea id="comment"></textarea>
<p><button id="next" disabled>Next</button></p>
</div>
</section>

<section id="finish" hidden>
<h2>Thank you</h2>
<p>Save your answers and send the file to the study.</p>
<button id="save">Save answers</button>
<p class="muted" id="saved"></p>
</section>
</main>
<script>
const CONFIG = __CONFIG__;
(function () {
  "use strict";
  const trials = CONFIG.trials;
  const answers = [];
  let index = 0;
  let chosen = null;
  const byId = (id) => document.getElementById(id);
  byId("min-age").textContent = String(CONFIG.minimumAge);

  function checkStart() {
    byId("start").disabled = !(
      byId("agree").checked && byId("age").checked && byId("hearing").checked
    );
  }
  ["agree", "age", "hearing"].forEach((id) => byId(id).addEventListener("change", checkStart));

  byId("start").addEventListener("click", function () {
    byId("consent").hidden = true;
    byId("trials").hidden = false;
    showTrial();
  });

  function showTrial() {
    const trial = trials[index];
    chosen = null;
    byId("progress").textContent = "Trial " + (index + 1) + " of " + trials.length;
    byId("clip-a").src = trial.a;
    byId("clip-b").src = trial.b;
    byId("comment").value = "";
    const box = byId("choices");
    box.textContent = "";
    CONFIG.answers.forEach(function (label) {
      const button = document.createElement("button");
      button.textContent = label;
      button.addEventListener("click", function () {
        chosen = label;
        Array.prototype.forEach.call(box.children, function (b) { b.classList.remove("chosen"); });
        button.classList.add("chosen");
        byId("next").disabled = false;
      });
      box.appendChild(button);
    });
    byId("next").disabled = true;
  }

  byId("next").addEventListener("click", function () {
    answers.push({
      trial_id: trials[index].id,
      answer: chosen,
      comment: byId("comment").value,
      answered_at: new Date().toISOString()
    });
    index += 1;
    if (index < trials.length) {
      showTrial();
    } else {
      byId("trials").hidden = true;
      byId("finish").hidden = false;
    }
  });

  byId("save").addEventListener("click", function () {
    const document_ = {
      format: CONFIG.format,
      protocol_sha256: CONFIG.protocolSha256,
      listener_id: CONFIG.listenerId,
      listener_index: CONFIG.listenerIndex,
      consent: CONFIG.consent,
      saved_at: new Date().toISOString(),
      answers: answers
    };
    const text = JSON.stringify(document_, null, 2) + "\\n";
    const blob = new Blob([text], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "answers_" + CONFIG.listenerId + ".json";
    document.body.appendChild(link);
    link.click();
    link.remove();
    byId("saved").textContent = "Saved " + answers.length + " answers.";
  });
})();
</script>
</body>
</html>
"""
