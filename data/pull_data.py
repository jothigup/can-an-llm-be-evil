"""Download a dataset file into the repository's local data directory."""

import argparse
import shutil
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen


def download_file(url: str, output: Path | None = None) -> Path:
    """Download a URL to a file, defaulting to data/raw/<URL filename>."""
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise ValueError("URL must be an absolute http or https URL")

    if output is None:
        filename = unquote(Path(parsed_url.path).name)
        if not filename:
            raise ValueError("URL has no filename; provide an output path")
        output = Path(__file__).parent / "raw" / filename

    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=30) as response, output.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="HTTP or HTTPS URL of the file to download")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="destination path (default: data/raw/<URL filename>)",
    )
    args = parser.parse_args()

    destination = download_file(args.url, args.output)
    print(f"Downloaded to {destination}")


if __name__ == "__main__":
    main()
