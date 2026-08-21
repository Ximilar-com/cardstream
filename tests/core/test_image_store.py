"""--store-images: one file per paid call, byte-identical to what was sent."""

from __future__ import annotations

import base64
import re

import pytest

from cardstream.core.image_store import FRAME, OBJECT, ImageStore

JPEG = b"\xff\xd8\xff\xe0 not a real jpeg, but exact bytes are the point \xff\xd9"
B64 = base64.b64encode(JPEG).decode("ascii")


def test_the_folder_is_created_up_front(tmp_path):
    folder = tmp_path / "shows" / "tonight"
    ImageStore(folder)
    assert folder.is_dir()


def test_an_uncreatable_folder_fails_at_construction(tmp_path):
    """A file where the folder should be — caught at startup, as a ValueError
    the entrypoints already turn into a clean `error:` exit."""
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("")
    with pytest.raises(ValueError, match="store-images"):
        ImageStore(blocker / "sub")


def test_what_lands_on_disk_is_what_was_sent(tmp_path):
    path = ImageStore(tmp_path).save_b64(B64)
    assert path is not None
    assert path.read_bytes() == JPEG


def test_names_carry_the_call_number_in_order(tmp_path):
    store = ImageStore(tmp_path)
    names = [store.save_b64(B64).name for _ in range(3)]
    assert [n[:5] for n in names] == ["00001", "00002", "00003"]
    assert all(re.fullmatch(r"\d{5}-[0-9a-f]{6}\.jpg", n) for n in names)
    assert sorted(names) == names  # the folder sorts in the order cards were shown


def test_a_second_run_does_not_overwrite_the_first(tmp_path):
    """Both runs start their numbering at 1; the random suffix keeps them apart."""
    first = ImageStore(tmp_path).save_b64(B64)
    second = ImageStore(tmp_path).save_b64(B64)
    assert first != second
    assert len(list(tmp_path.iterdir())) == 2


def test_bad_base64_is_logged_and_skipped_not_raised(tmp_path, caplog):
    store = ImageStore(tmp_path)
    assert store.save_b64("this is not base64!") is None
    assert not list(tmp_path.iterdir())
    assert "not saved" in caplog.text


def test_a_failing_write_costs_the_archive_not_the_call(tmp_path, monkeypatch, caplog):
    """A full disk mid-show must not propagate into the identify thread."""
    store = ImageStore(tmp_path)

    def boom(self, data):
        raise OSError("No space left on device")

    monkeypatch.setattr("pathlib.Path.write_bytes", boom)
    assert store.save_b64(B64) is None
    assert "No space left" in caplog.text


# --- --store-images-type -----------------------------------------------------


def _frame():
    import numpy as np

    return np.random.RandomState(0).randint(0, 256, size=(48, 64, 3), dtype=np.uint8)


def test_object_mode_keeps_crops_and_ignores_frames(tmp_path):
    store = ImageStore(tmp_path, OBJECT)
    assert store.save_frame(_frame()) is None
    assert store.save_b64(B64) is not None
    assert len(list(tmp_path.iterdir())) == 1


def test_frame_mode_keeps_frames_and_ignores_crops(tmp_path):
    store = ImageStore(tmp_path, FRAME)
    assert store.save_b64(B64) is None
    assert store.save_frame(_frame()) is not None
    assert len(list(tmp_path.iterdir())) == 1


def test_frame_mode_writes_a_readable_jpeg_of_the_whole_frame(tmp_path):
    import cv2

    frame = _frame()
    path = ImageStore(tmp_path, FRAME).save_frame(frame)
    decoded = cv2.imread(str(path))
    assert decoded.shape == frame.shape  # the frame, not a crop of it


def test_object_is_the_default_mode(tmp_path):
    assert ImageStore(tmp_path).kind == OBJECT


def test_an_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="store-images-type"):
        ImageStore(tmp_path, "everything")


def test_the_numbering_is_one_sequence_whichever_mode(tmp_path):
    store = ImageStore(tmp_path, FRAME)
    names = [store.save_frame(_frame()).name for _ in range(2)]
    assert [n[:5] for n in names] == ["00001", "00002"]
