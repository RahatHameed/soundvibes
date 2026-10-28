#!/usr/bin/env python3
"""Read the English translation of a meeting aloud, as it is written.

Follows translate.EN.txt and speaks each new line on a separate output (the
laptop speakers), never the sink soundvibes is capturing, so it cannot hear
itself. Only lines whose original was German are spoken.

Gate: speech plays only while every microphone the browser is capturing is
muted at the system level. Unmuting one cuts the current sentence at once.
If the state cannot be read, nothing is spoken (fail closed).

Started by `make meeting`; needs `make voice` once.
"""

import argparse
import json
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
import wave

MAX_BACKLOG = 3  # drop older lines rather than fall behind the meeting
POLL_SECONDS = 0.15  # how often the mute gate is re-checked while speaking
LINE = re.compile(r"^\[([\d:]+)\] \[(\w+)\] \[(\w+)\] (.+)$")


def mics_muted() -> bool:
    """True only if every source feeding a browser input stream is muted.

    No browser input stream means no mic reaches the meeting, so that is safe.
    Any error means "not safe"."""
    try:
        dump = json.loads(
            subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=3).stdout
        )
        nodes = {
            o["id"]: o["info"]["props"] for o in dump if o.get("type") == "PipeWire:Interface:Node"
        }
        sources = {
            o["info"]["output-node-id"]
            for o in dump
            if o.get("type") == "PipeWire:Interface:Link"
            and nodes.get(o["info"]["input-node-id"], {}).get("media.class") == "Stream/Input/Audio"
            and "chrome" in nodes[o["info"]["input-node-id"]].get("application.name", "").lower()
        }
        for source in sources:
            out = subprocess.run(
                ["wpctl", "get-volume", str(source)], capture_output=True, text=True, timeout=3
            ).stdout
            if "[MUTED]" not in out:
                return False
        return True
    except Exception:
        return False


def find_speakers() -> str | None:
    """The built-in speakers: an audio sink described as "Speaker" that is not
    the default sink (the default is what the meeting plays on, and captured)."""
    dump = json.loads(subprocess.run(["pw-dump"], capture_output=True, text=True).stdout)
    default = next(
        (
            entry["value"].get("name")
            for o in dump
            if o.get("type") == "PipeWire:Interface:Metadata"
            for entry in o.get("metadata", [])
            if entry.get("key") == "default.audio.sink"
        ),
        None,
    )
    for o in dump:
        props = o.get("info", {}).get("props", {})
        if (
            props.get("media.class") == "Audio/Sink"
            and "speaker" in props.get("node.description", "").lower()
            and props.get("node.name") != default
        ):
            return props["node.name"]
    return None


def spoken_in_german(transcript: str, stamp: str, source: str) -> bool:
    """Was the original of this line German? The original is written before its
    translation, so it is already in the transcript. Unknown means don't speak."""
    try:
        with open(transcript, encoding="utf-8") as f:
            tail = f.readlines()[-200:]
    except OSError:
        return False
    languages = {
        m.group(3)
        for m in map(LINE.match, map(str.strip, tail))
        if m and m.group(1) == stamp and m.group(2) == source
    }
    return languages == {"de"}


def follow(translation: str, transcript: str, lines: "queue.Queue[str]") -> None:
    while not os.path.exists(translation):
        time.sleep(0.5)
    with open(translation, encoding="utf-8") as f:
        f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = LINE.match(line.strip())
            if m and spoken_in_german(transcript, m.group(1), m.group(2)):
                lines.put(m.group(4).rstrip(". …"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--transcript", default="transcripts/transcript.txt")
    parser.add_argument("--translation", default="transcripts/translate.EN.txt")
    parser.add_argument("--voice", required=True, help="piper .onnx voice model")
    parser.add_argument("--sink", help="PipeWire node name to speak on (default: the speakers)")
    args = parser.parse_args()
    args.sink = args.sink or find_speakers()
    if not args.sink:
        parser.error("no built-in speakers found; pass --sink (see `wpctl status`)")

    from piper import PiperVoice  # heavy; only after the arguments are valid

    voice = PiperVoice.load(args.voice)
    lines: queue.Queue[str] = queue.Queue()
    threading.Thread(
        target=follow, args=(args.translation, args.transcript, lines), daemon=True
    ).start()

    # make sends SIGTERM on shutdown; turn it into a normal exit so the
    # current sentence is cut instead of playing on after we are gone.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))

    print(f"Speaking German->English on {args.sink}", flush=True)
    wav_path = os.path.join(tempfile.gettempdir(), f"speak-{os.getpid()}.wav")
    player = None
    gated = False
    try:
        while True:
            text = lines.get()
            while lines.qsize() >= MAX_BACKLOG:
                text = lines.get()
            if not mics_muted():
                if not gated:
                    print("!! mic is UNMUTED - reading paused until it is muted", flush=True)
                    gated = True
                continue
            if gated:
                print("mic muted - reading resumed", flush=True)
                gated = False
            print(f"> {text}", flush=True)
            with wave.open(wav_path, "wb") as w:
                voice.synthesize_wav(text, w)
            player = subprocess.Popen(["pw-play", "--target", args.sink, wav_path])
            while player.poll() is None:
                if not mics_muted():
                    player.kill()
                    print("!! mic UNMUTED - speech cut", flush=True)
                    gated = True
                    break
                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        pass
    finally:
        if player is not None and player.poll() is None:
            player.kill()
        if os.path.exists(wav_path):
            os.remove(wav_path)


if __name__ == "__main__":
    main()
