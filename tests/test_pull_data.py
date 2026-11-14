from io import BytesIO

import pytest

from data.pull_data import download_file


class FakeResponse(BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def test_download_file_uses_filename_from_url(tmp_path, monkeypatch):
    response = FakeResponse(b"dataset contents")
    monkeypatch.setattr("data.pull_data.urlopen", lambda url, timeout: response)
    monkeypatch.setattr("data.pull_data.__file__", str(tmp_path / "pull_data.py"))

    destination = download_file("https://example.test/datasets/sample.csv")

    assert destination == tmp_path / "raw" / "sample.csv"
    assert destination.read_bytes() == b"dataset contents"


def test_download_file_accepts_explicit_output(tmp_path, monkeypatch):
    response = FakeResponse(b"dataset contents")
    monkeypatch.setattr("data.pull_data.urlopen", lambda url, timeout: response)
    destination = tmp_path / "custom" / "data.csv"

    assert download_file("https://example.test/data", destination) == destination
    assert destination.read_bytes() == b"dataset contents"


@pytest.mark.parametrize("url", ["file:///tmp/data.csv", "/relative/path"])
def test_download_file_rejects_non_http_urls(url):
    with pytest.raises(ValueError, match="http or https"):
        download_file(url)


def test_download_file_requires_filename_or_output():
    with pytest.raises(ValueError, match="no filename"):
        download_file("https://example.test/")
