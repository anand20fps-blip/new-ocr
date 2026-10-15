"""Windows Scan text, plus the characters that English Scan text drops.

Scan text stays the reading of the page. A second pass only fills in characters
English cannot keep: the copyright sign, Latin accents, Turkish letters, and
Japanese, Chinese, Korean, and Tamil.
"""

from __future__ import annotations

import shutil
import subprocess
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TESSDATA = _ROOT / "tessdata"
_TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Fast models for the scripts named for this checker, plus the Latin script
# model that keeps accents. English Scan text remains the base reading.
_MODELS = {
    "Latin.traineddata": "https://github.com/tesseract-ocr/tessdata/raw/main/script/Latin.traineddata",
    "tur.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/tur.traineddata",
    "jpn.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/jpn.traineddata",
    "chi_sim.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_sim.traineddata",
    "chi_tra.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/chi_tra.traineddata",
    "kor.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/kor.traineddata",
    "tam.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/tam.traineddata",
}
_LATIN_LANGS = "Latin+tur"
_SCRIPT_LANGS = "jpn+chi_sim+chi_tra+kor+tam"
_TURKISH = set("\u011f\u011e\u015f\u015e\u0131\u0130")
_COPY_TOKENS = {"0", "O", "o", "Q"}


@dataclass
class Word:
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float = 100.0


@dataclass
class Line:
    words: list[Word]


def ensure_models() -> Path | None:
    """Make the script models available. Returns tessdata dir, or None."""
    exe = _tesseract()
    if exe is None:
        print("Script supplement off: Tesseract was not found.", flush=True)
        return None
    _TESSDATA.mkdir(parents=True, exist_ok=True)
    eng_src = Path(r"C:\Program Files\Tesseract-OCR\tessdata\eng.traineddata")
    eng_dst = _TESSDATA / "eng.traineddata"
    if eng_src.is_file() and not eng_dst.is_file():
        shutil.copy2(eng_src, eng_dst)
    _copy_tess_configs()
    for name, url in _MODELS.items():
        dest = _TESSDATA / name
        if dest.is_file() and dest.stat().st_size > 100_000:
            continue
        print(f"download {name}", flush=True)
        urllib.request.urlretrieve(url, dest)
    missing = [name for name in _MODELS if not (_TESSDATA / name).is_file()]
    if missing:
        print("Script supplement off, missing: " + ", ".join(missing), flush=True)
        return None
    print(
        "Script supplement: Latin accents, Turkish, Japanese, "
        "Chinese, Korean, Tamil",
        flush=True,
    )
    return _TESSDATA


def _copy_tess_configs() -> None:
    """TSV output needs the config files that ship with Tesseract."""
    source = Path(r"C:\Program Files\Tesseract-OCR\tessdata")
    for folder in ("configs", "tessconfigs"):
        src = source / folder
        dest = _TESSDATA / folder
        if src.is_dir() and not dest.exists():
            shutil.copytree(src, dest)


def _tesseract() -> Path | None:
    if _TESSERACT.is_file():
        return _TESSERACT
    found = shutil.which("tesseract")
    return Path(found) if found else None


def apply_character_repairs(image_path: Path, lines: list[Line], tessdata: Path | None) -> str:
    """Repair copyright marks and fill scripts Scan text missed. Returns page text."""
    _repair_copyright(image_path, lines)
    if tessdata is not None:
        _merge_special_words(lines, _read_tsv(image_path, tessdata, _LATIN_LANGS), latin=True)
        _merge_special_words(lines, _read_tsv(image_path, tessdata, _SCRIPT_LANGS), latin=False)
    return _page_text(lines)


def _page_text(lines: list[Line]) -> str:
    ordered = _reading_order(lines)
    parts: list[str] = []
    for line in ordered:
        text = _join_line(line.words)
        text = text.strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip()


def _reading_order(lines: list[Line]) -> list[Line]:
    """Read a two- or three-column page down the left column, then the next.

    A line that stretches across the text width stays where it is, so a
    running header or footer is not pulled into a column. A one-column page
    is unchanged.
    """
    usable = [line for line in lines if line.words]
    if len(usable) < 6:
        return _by_y(usable)
    usable = _merge_line_fragments(usable)
    span = _text_span(usable)
    if span < 1:
        return _by_y(usable)
    full = [line for line in usable if _line_width(line) >= 0.72 * span]
    body = [line for line in usable if line not in full]
    columns = _split_columns(body, span)
    if len(columns) < 2:
        return _by_y(usable)
    column_top = min(_line_y(line) for column in columns for line in column)
    above = [line for line in full if _line_y(line) < column_top - 8]
    below = [line for line in full if line not in above]
    ordered = _by_y(above)
    for column in columns:
        ordered.extend(_by_y(column))
    ordered.extend(_by_y(below))
    return ordered


def _split_columns(lines: list[Line], span: float) -> list[list[Line]]:
    if len(lines) < 6:
        return [lines]
    ordered = sorted(lines, key=_line_x0)
    gap_min = span * 0.18
    clusters: list[list[Line]] = [[ordered[0]]]
    for previous, line in zip(ordered, ordered[1:]):
        if _line_x0(line) - _line_x0(previous) >= gap_min:
            clusters.append([line])
        else:
            clusters[-1].append(line)
    columns = [cluster for cluster in clusters if len(cluster) >= 3]
    if len(columns) < 2:
        return [lines]
    centers = [sum(_line_x0(line) for line in column) / len(column) for column in columns]
    for cluster in clusters:
        if len(cluster) >= 3:
            continue
        for line in cluster:
            nearest = min(range(len(columns)), key=lambda index: abs(_line_x0(line) - centers[index]))
            columns[nearest].append(line)
    columns.sort(key=lambda column: min(_line_x0(line) for line in column))
    return columns


def _merge_line_fragments(lines: list[Line]) -> list[Line]:
    """Join a wrapped tail that Scan text split off the end of the same line."""
    span = _text_span(lines)
    parent = list(range(len(lines)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    # A wrapped tail sits a short gap past its own line. The next column
    # starts much farther from that line's left edge, so it is not a tail.
    gap_limit = 32.0
    same_column = span * 0.48
    for i, left in enumerate(lines):
        for j in range(i + 1, len(lines)):
            right = lines[j]
            pair = (left, right) if _line_x0(left) <= _line_x0(right) else (right, left)
            if not _same_band(pair[0], pair[1]):
                continue
            gap = _line_x0(pair[1]) - _line_x1(pair[0])
            if gap < 0 or gap > gap_limit:
                continue
            if _line_x0(pair[1]) - _line_x0(pair[0]) > same_column:
                continue
            parent[find(j)] = find(i)
    groups: dict[int, list[Word]] = {}
    for index, line in enumerate(lines):
        groups.setdefault(find(index), []).extend(line.words)
    merged = [Line(sorted(words, key=lambda word: word.x)) for words in groups.values()]
    return merged


def _by_y(lines: list[Line]) -> list[Line]:
    return sorted(lines, key=_line_y)


def _text_span(lines: list[Line]) -> float:
    if not lines:
        return 0.0
    return max(_line_x1(line) for line in lines) - min(_line_x0(line) for line in lines)


def _line_x0(line: Line) -> float:
    return min(word.x for word in line.words)


def _line_x1(line: Line) -> float:
    return max(word.x + word.w for word in line.words)


def _line_y(line: Line) -> float:
    return min(word.y for word in line.words)


def _line_width(line: Line) -> float:
    return _line_x1(line) - _line_x0(line)


def _line_height(line: Line) -> float:
    return max(word.h for word in line.words)


def _same_band(left: Line, right: Line) -> bool:
    band = 0.55 * max(_line_height(left), _line_height(right), 8)
    return abs(_line_y(left) - _line_y(right)) <= band


def _join_line(words: list[Word]) -> str:
    texts = [_clean_word(word.text) for word in words]
    texts = [text for text in texts if text]
    letters = [ch for text in texts for ch in text if ch.isalpha()]
    if letters and all(_script(ch) in {"han", "kana"} for ch in letters):
        return "".join(texts)
    return " ".join(texts)


def _clean_word(text: str) -> str:
    kept = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return unicodedata.normalize("NFC", kept).strip()


def _repair_copyright(image_path: Path, lines: list[Line]) -> None:
    import pymupdf

    pixmap = pymupdf.Pixmap(str(image_path))
    flat = [word for line in lines for word in line.words]
    for index, word in enumerate(flat):
        token = word.text.strip()
        if token in {"\u00a9", "(c)", "(C)"}:
            word.text = "\u00a9"
            continue
        if token not in _COPY_TOKENS:
            continue
        previous = flat[index - 1].text if index else ""
        following = flat[index + 1].text if index + 1 < len(flat) else ""
        if _copyright_context(previous, following) or _circled_c(pixmap, word):
            word.text = "\u00a9"


def _copyright_context(previous: str, following: str) -> bool:
    prev = previous.strip(" .:;").casefold()
    if prev != "copyright":
        return False
    year = following.strip(" .,;:")
    return len(year) >= 4 and year[:4].isdigit()


def _circled_c(pixmap, word: Word) -> bool:
    """A circled C is square and darker on the left of the hole than the right."""
    if word.w < 12 or word.h < 12:
        return False
    if not 0.85 <= word.w / word.h <= 1.2:
        return False
    samples = pixmap.samples
    channels = pixmap.n
    width = pixmap.width
    height = pixmap.height
    x0 = max(0, word.x)
    y0 = max(0, word.y)
    x1 = min(width, word.x + word.w)
    y1 = min(height, word.y + word.h)
    box_w = x1 - x0
    box_h = y1 - y0
    if box_w < 12 or box_h < 12:
        return False

    def dark(x: int, y: int) -> bool:
        offset = (y * width + x) * channels
        if channels == 1:
            gray = samples[offset]
        else:
            gray = (samples[offset] + samples[offset + 1] + samples[offset + 2]) // 3
        return gray < 160

    left = left_dark = right = right_dark = rim = rim_dark = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            local_x = x - x0
            local_y = y - y0
            ink = dark(x, y)
            on_rim = (
                local_x < box_w * 0.18
                or local_x > box_w * 0.82
                or local_y < box_h * 0.18
                or local_y > box_h * 0.82
            )
            if on_rim:
                rim += 1
                if ink:
                    rim_dark += 1
            if not box_h * 0.25 < local_y < box_h * 0.75:
                continue
            if box_w * 0.15 < local_x < box_w * 0.48:
                left += 1
                if ink:
                    left_dark += 1
            elif box_w * 0.52 < local_x < box_w * 0.85:
                right += 1
                if ink:
                    right_dark += 1
    if left == 0 or right == 0 or rim == 0:
        return False
    left_ratio = left_dark / left
    right_ratio = right_dark / right
    # The C sits on the left of the circle, so the hole is darker on the left,
    # while the ring still carries ink around the outside.
    return (
        left_ratio >= 0.25
        and left_ratio >= right_ratio * 1.7
        and rim_dark / rim >= 0.2
    )


def _read_tsv(image_path: Path, tessdata: Path, langs: str) -> list[Word]:
    exe = _tesseract()
    if exe is None:
        return []
    try:
        proc = subprocess.run(
            [
                str(exe),
                str(image_path),
                "stdout",
                "--tessdata-dir",
                str(tessdata),
                "-l",
                langs,
                "--oem",
                "1",
                "--psm",
                "3",
                "tsv",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    words: list[Word] = []
    text = proc.stdout.decode("utf-8", errors="replace")
    for row in text.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12 or cols[0] != "5":
            continue
        token = cols[11].strip()
        if not token:
            continue
        try:
            conf = float(cols[10])
            left, top, width, height = (int(float(cols[i])) for i in range(6, 10))
        except ValueError:
            continue
        if conf < 0 or width <= 0 or height <= 0:
            continue
        words.append(Word(unicodedata.normalize("NFC", token), left, top, width, height, conf))
    return words


def _merge_special_words(lines: list[Line], extra: list[Word], *, latin: bool) -> None:
    if not extra:
        return
    if latin:
        _merge_latin_words(lines, extra)
    else:
        _merge_script_lines(lines, extra)


def _merge_latin_words(lines: list[Line], extra: list[Word]) -> None:
    placed = [word for line in lines for word in line.words]
    for candidate in extra:
        if candidate.conf < 80 or not _has_latin_extension(candidate.text):
            continue
        host = _best_overlap(candidate, placed)
        if host is None:
            continue
        same_word = _skeleton(host.text) == _skeleton(candidate.text)
        same_ink = _overlap_ratio(candidate, host) >= 0.85
        if host.text == candidate.text:
            continue
        if same_word and _mark_count(candidate.text) >= _mark_count(host.text):
            host.text = candidate.text
        elif same_ink and _mark_count(candidate.text) > _mark_count(host.text):
            host.text = candidate.text


def _merge_script_lines(lines: list[Line], extra: list[Word]) -> None:
    """Add Japanese, Chinese, Korean, and Tamil lines Scan text skipped.

    One false letter on an English page is ignored. A real script line has
    several characters, and that whole line is kept in reading order.
    """
    strong = [word for word in extra if word.conf >= 85 and _non_latin_script(word.text)]
    if sum(_script_letters(word.text) for word in strong) < 3:
        return
    for cluster in _cluster_words(strong):
        host = _line_near(lines, cluster)
        ordered = sorted(cluster, key=lambda word: word.x)
        if host is None:
            lines.append(Line(ordered))
            continue
        if any(_non_latin_script(word.text) for word in host.words):
            continue
        latin_letters = sum(
            1 for word in host.words for ch in word.text if ch.isascii() and ch.isalpha()
        )
        script_letters = sum(_script_letters(word.text) for word in cluster)
        if latin_letters >= 8 and script_letters < latin_letters:
            continue
        host.words = ordered


def _cluster_words(words: list[Word]) -> list[list[Word]]:
    ordered = sorted(words, key=lambda word: (word.y, word.x))
    clusters: list[list[Word]] = []
    for word in ordered:
        center = word.y + word.h / 2
        if clusters:
            previous = clusters[-1]
            prev_center = sum(item.y + item.h / 2 for item in previous) / len(previous)
            limit = max(word.h, previous[-1].h, 8) * 0.7
            if abs(center - prev_center) <= limit:
                previous.append(word)
                continue
        clusters.append([word])
    return clusters


def _line_near(lines: list[Line], cluster: list[Word]) -> Line | None:
    center = sum(word.y + word.h / 2 for word in cluster) / len(cluster)
    height = max(word.h for word in cluster)
    best: Line | None = None
    best_distance = height * 0.8
    for line in lines:
        if not line.words:
            continue
        line_center = sum(word.y + word.h / 2 for word in line.words) / len(line.words)
        distance = abs(center - line_center)
        if distance <= best_distance:
            best = line
            best_distance = distance
    return best


def _best_overlap(candidate: Word, hosts: list[Word]) -> Word | None:
    best: Word | None = None
    best_ratio = 0.45
    for host in hosts:
        ratio = _overlap_ratio(candidate, host)
        if ratio > best_ratio:
            best = host
            best_ratio = ratio
    return best


def _overlap_ratio(a: Word, b: Word) -> float:
    x0 = max(a.x, b.x)
    y0 = max(a.y, b.y)
    x1 = min(a.x + a.w, b.x + b.w)
    y1 = min(a.y + a.h, b.y + b.h)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    smaller = min(a.w * a.h, b.w * b.h)
    if smaller <= 0:
        return 0.0
    return inter / smaller


def _skeleton(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    bare = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    folded = bare.casefold().translate(
        str.maketrans({"\u0131": "i", "\u015f": "s", "\u011f": "g"})
    )
    return "".join(ch for ch in folded if ch.isalnum())


def _mark_count(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() and ord(ch) > 127)


def _script_letters(text: str) -> int:
    return sum(1 for ch in text if _script(ch) != "other")


def _has_latin_extension(text: str) -> bool:
    for ch in text:
        if ch in _TURKISH:
            return True
        if not ch.isalpha() or ord(ch) < 128:
            continue
        if unicodedata.name(ch, "").startswith("LATIN"):
            return True
    return False


def _non_latin_script(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    special = sum(1 for ch in letters if _script(ch) in {"han", "kana", "hangul", "tamil"})
    return special / len(letters) >= 0.6


def _script(ch: str) -> str:
    code = ord(ch)
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return "kana"
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF:
        return "han"
    if 0xAC00 <= code <= 0xD7AF or 0x1100 <= code <= 0x11FF:
        return "hangul"
    if 0x0B80 <= code <= 0x0BFF:
        return "tamil"
    return "other"


async def recognize(image_path: Path, engine, tessdata: Path | None) -> str:
    """Read one page image with Scan text, then repair special characters."""
    from winsdk.windows.graphics.imaging import BitmapDecoder
    from winsdk.windows.storage import FileAccessMode, StorageFile

    storage_file = await StorageFile.get_file_from_path_async(str(image_path.resolve()))
    stream = await storage_file.open_async(FileAccessMode.READ)
    try:
        decoder = await BitmapDecoder.create_async(stream)
        bitmap = await decoder.get_software_bitmap_async()
        result = await engine.recognize_async(bitmap)
    finally:
        stream.close()
    lines = _lines_from_result(result)
    if not lines:
        return (result.text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return apply_character_repairs(image_path, lines, tessdata)


def _lines_from_result(result) -> list[Line]:
    lines: list[Line] = []
    for line in result.lines:
        words: list[Word] = []
        for word in line.words:
            rect = word.bounding_rect
            text = (word.text or "").strip()
            if not text:
                continue
            words.append(
                Word(
                    text,
                    int(rect.x),
                    int(rect.y),
                    max(1, int(rect.width)),
                    max(1, int(rect.height)),
                )
            )
        if words:
            lines.append(Line(words))
    return lines


async def scan_engine():
    from winsdk.windows.globalization import Language
    from winsdk.windows.media.ocr import OcrEngine

    engine = OcrEngine.try_create_from_user_profile_languages()
    if engine is not None:
        return engine
    english = Language("en-US")
    if OcrEngine.is_language_supported(english):
        return OcrEngine.try_create_from_language(english)
    raise RuntimeError(
        "Windows Scan text has no OCR language installed. "
        "Add English (or another language) with Optical character recognition "
        "under Settings > Time & language > Language & region."
    )
