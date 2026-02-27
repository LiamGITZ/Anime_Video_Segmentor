from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


YOUTUBE_URL_RE = re.compile(
    r"^(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[A-Za-z0-9_-]{6,}([&?].*)?$"
)


def _validate_youtube_url(url: str) -> str:
    if not YOUTUBE_URL_RE.match(url):
        raise argparse.ArgumentTypeError(
            "Please provide a valid YouTube video URL (youtube.com/watch?v=... or youtu.be/...)."
        )
    return url


def download_audio(youtube_url: str, output_dir: Path) -> Path:
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: yt-dlp. Install it with: pip install yt-dlp"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(title)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        "no_warnings": True,
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        title = info.get("title", "downloaded_audio")

    return output_dir / f"{title}.mp3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a YouTube video's audio track into output_song."
    )
    parser.add_argument("url", type=_validate_youtube_url, help="YouTube video URL")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("output_song"),
        help="Output directory (default: output_song)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        audio_path = download_audio(args.url, args.output_dir)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved audio to: {audio_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
