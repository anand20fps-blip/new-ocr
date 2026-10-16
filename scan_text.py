"""Windows Scan text, plus the characters that English Scan text drops.

Scan text stays the reading of the page. A second pass fills in what English
drops: the copyright sign, any accented Latin letter Unicode defines, Turkish
letters, and Japanese, Chinese, Korean, Tamil, Thai, Arabic, Malayalam, and
Telugu. Quotes, dashes, currency, and other symbols are kept when the scan reads them.
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
    "tha.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/tha.traineddata",
    "ara.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/ara.traineddata",
    "mal.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/mal.traineddata",
    "tel.traineddata": "https://github.com/tesseract-ocr/tessdata_fast/raw/main/tel.traineddata",
}
_LATIN_LANGS = "Latin+tur"
_SCRIPT_LANGS = "jpn+chi_sim+chi_tra+kor+tam+tha+ara+mal+tel"
_SCRIPTS = {"han", "kana", "hangul", "tamil", "thai", "arabic", "malayalam", "telugu"}
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


@dataclass
class PlacedLine:
    text: str
    x: int
    y: int
    w: int
    h: int


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
        "Script supplement: Latin accents, Turkish, Japanese, Chinese, "
        "Korean, Tamil, Thai, Arabic, Malayalam, Telugu",
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
        _repair_printed_accents(image_path, lines, tessdata)
    return _page_text(lines)


def _ordered_lines(lines: list[Line]) -> list[Line]:
    return sorted(
        (line for line in lines if line.words),
        key=lambda line: (min(word.y for word in line.words), min(word.x for word in line.words)),
    )


def placed_lines(lines: list[Line]) -> list[PlacedLine]:
    """One entry per visual line, with its box in image pixels."""
    placed: list[PlacedLine] = []
    for line in _ordered_lines(lines):
        text = _join_line(line.words).strip()
        if not text:
            continue
        x0 = min(word.x for word in line.words)
        y0 = min(word.y for word in line.words)
        x1 = max(word.x + word.w for word in line.words)
        y1 = max(word.y + word.h for word in line.words)
        placed.append(PlacedLine(text, x0, y0, max(1, x1 - x0), max(1, y1 - y0)))
    return placed


def _page_text(lines: list[Line]) -> str:
    parts = [line.text for line in placed_lines(lines)]
    return "\n".join(parts).strip()


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


def _ink_rows(image_path: Path) -> tuple[list[bytearray], int]:
    import pymupdf

    pixmap = pymupdf.Pixmap(str(image_path))
    raw = pixmap.samples
    channels = pixmap.n
    width = pixmap.width
    rows: list[bytearray] = []
    for y in range(pixmap.height):
        base = y * pixmap.stride
        row = bytearray(width)
        if channels == 1:
            line = raw[base : base + width]
            for x, gray in enumerate(line):
                if gray < 150:
                    row[x] = 1
        else:
            line = raw[base : base + width * channels]
            for x in range(width):
                pixel = x * channels
                gray = (line[pixel] + line[pixel + 1] + line[pixel + 2]) // 3
                if gray < 150:
                    row[x] = 1
        rows.append(row)
    return rows, pixmap.height


def _char_boxes(image_path: Path, tessdata: Path, height: int) -> list[tuple[str, int, int, int, int]]:
    """Character boxes from Tesseract, in top-left image pixels."""
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
                _LATIN_LANGS,
                "--oem",
                "1",
                "--psm",
                "3",
                "makebox",
            ],
            capture_output=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode != 0:
        return []
    boxes: list[tuple[str, int, int, int, int]] = []
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 5 or len(parts[0]) != 1:
            continue
        left, bottom, right, top = (int(float(parts[i])) for i in range(1, 5))
        boxes.append((parts[0], left, height - top, right, height - bottom))
    return boxes


def _accent_blobs(mask: list[list[int]]) -> list[list[tuple[int, int]]]:
    if not mask or not mask[0]:
        return []
    height, width = len(mask), len(mask[0])
    seen = [[False] * width for _ in range(height)]
    blobs: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if not mask[y][x] or seen[y][x]:
                continue
            stack = [(x, y)]
            seen[y][x] = True
            points: list[tuple[int, int]] = []
            while stack:
                cx, cy = stack.pop()
                points.append((cx, cy))
                for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                    if 0 <= nx < width and 0 <= ny < height and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            if len(points) >= 4:
                blobs.append(points)
    return blobs


def _has_hole(points: list[tuple[int, int]]) -> bool:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    width, height = x1 - x0 + 1, y1 - y0 + 1
    if width < 5 or height < 5:
        return False
    ink = [[False] * width for _ in range(height)]
    for x, y in points:
        ink[y - y0][x - x0] = True
    seen = [[False] * width for _ in range(height)]
    stack = []
    for x in range(width):
        stack.append((x, 0))
        stack.append((x, height - 1))
    for y in range(height):
        stack.append((0, y))
        stack.append((width - 1, y))
    while stack:
        x, y = stack.pop()
        if not 0 <= x < width or not 0 <= y < height or seen[y][x] or ink[y][x]:
            continue
        seen[y][x] = True
        stack.extend(((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)))
    return any(not ink[y][x] and not seen[y][x] for y in range(height) for x in range(width))


def _blob_size(points: list[tuple[int, int]]) -> tuple[int, int]:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return max(xs) - min(xs) + 1, max(ys) - min(ys) + 1


def _is_dot(points: list[tuple[int, int]]) -> bool:
    """A round mark. A short diagonal stroke is square too, but it has a slope."""
    width, height = _blob_size(points)
    fill = len(points) / max(1, width * height)
    compact = max(width, height) <= 1.6 * min(width, height) and width <= 8 and fill >= 0.4
    return compact and abs(_slope(points)) < 0.35


def _slope(points: list[tuple[int, int]]) -> float:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    variance = sum((x - mean_x) ** 2 for x in xs)
    if variance < 1:
        return 0.0
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / variance


def _tops(points: list[tuple[int, int]]) -> list[int] | None:
    """Highest ink in each column. Image y grows downward, so a smaller value is higher."""
    by_x: dict[int, int] = {}
    for x, y in points:
        by_x[x] = y if x not in by_x else min(by_x[x], y)
    xs = sorted(by_x)
    if len(xs) < 4:
        return None
    return [by_x[x] for x in xs]


def _outline_mark(points: list[tuple[int, int]]) -> str | None:
    """Circumflex, caron, breve, or tilde, from the top edge of one accent."""
    tops = _tops(points)
    if not tops:
        return None
    width, height = _blob_size(points)
    if width < height:
        return None
    third = max(1, len(tops) // 3)
    left = sum(tops[:third]) / third
    mid_slice = tops[third : len(tops) - third] or tops
    mid = sum(mid_slice) / len(mid_slice)
    right = sum(tops[-third:]) / third
    # Two direction changes along the top edge: a tilde.
    signs: list[int] = []
    for previous, current in zip(tops, tops[1:]):
        sign = 1 if current > previous + 0 else (-1 if current < previous else 0)
        if sign and (not signs or signs[-1] != sign):
            signs.append(sign)
    if len(signs) >= 3 and width >= 2 * height:
        return "\u0303"
    peak = min(left, right) - mid
    valley = mid - max(left, right)
    if peak >= 0.4 * max(height, 1) and mid < left and mid < right:
        return "\u0302"
    if valley >= 0.4 * max(height, 1) and mid > left and mid > right:
        # A caron comes to a point. A breve is the same arch, but shallow.
        return "\u030c" if valley >= 0.45 * max(height, 1) else "\u0306"
    return None


def _flat_bar(points: list[tuple[int, int]]) -> bool:
    tops = _tops(points)
    if not tops:
        return False
    return max(tops) - min(tops) <= 1


def _two_column_dots(mask: list[list[int]]) -> str | None:
    """Two side-by-side dots. A single stroke has no empty column between its ink."""
    # A circumflex is taller than a pair of dots and must not be split into two.
    if not mask or not mask[0] or len(mask) > 4:
        return None
    width = len(mask[0])
    cols = [sum(row[x] for row in mask) for x in range(width)]
    groups: list[tuple[int, int]] = []
    start = None
    for index, count in enumerate(cols + [0]):
        if count:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= 2 and sum(cols[start:index]) >= 4:
                groups.append((start, index))
            start = None
    if len(groups) < 2:
        return None
    left, right = groups[0], groups[1]
    if right[0] - left[1] < 2:
        return None
    if left[1] - left[0] > 8 or right[1] - right[0] > 8:
        return None
    return "\u0308"


def _mark_from_blobs(blobs: list[list[tuple[int, int]]]) -> str | None:
    if len(blobs) >= 2:
        pair = sorted(blobs, key=len, reverse=True)[:2]
        first, second = pair
        y1 = sum(point[1] for point in first) / len(first)
        y2 = sum(point[1] for point in second) / len(second)
        x1 = sum(point[0] for point in first) / len(first)
        x2 = sum(point[0] for point in second) / len(second)
        side_by_side = abs(y1 - y2) <= 4 and abs(x1 - x2) >= 3
        if (
            side_by_side
            and _is_dot(first)
            and _is_dot(second)
            and _blob_size(first)[1] <= 4
            and _blob_size(second)[1] <= 4
        ):
            return "\u0308"
        if side_by_side and _slope(first) <= -0.2 and _slope(second) <= -0.2:
            return "\u030b"
    points = max(blobs, key=len)
    width, height = _blob_size(points)
    if _has_hole(points) and 0.6 <= width / max(height, 1) <= 1.7:
        return "\u030a"
    # A peaked accent is a circumflex or caron. A flat bar is a macron.
    outlined = _outline_mark(points)
    if outlined:
        return outlined
    if width >= 1.6 * height and height <= 4 and _flat_bar(points):
        return "\u0304"
    slope = _slope(points)
    if _is_dot(points):
        return None
    if slope <= -0.2:
        return "\u0301"
    if slope >= 0.2:
        return "\u0300"
    return None


def _printed_mark(rows: list[bytearray], box: tuple[int, int, int, int]) -> str | None:
    """The accent floating above this glyph, read from its shape."""
    mark = _printed_mark_in(rows, box)
    if mark:
        return mark
    # A short acute is often clipped on the right edge of the character box.
    x0, y0, x1, y1 = box
    return _printed_mark_in(rows, (x0, y0, x1 + 6, y1))


def _printed_mark_in(rows: list[bytearray], box: tuple[int, int, int, int]) -> str | None:
    height, width = len(rows), len(rows[0])
    x0 = max(0, box[0])
    y0 = max(0, box[1])
    x1 = min(width, box[2])
    y1 = min(height, box[3])
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    counts = [sum(rows[y][x0:x1]) for y in range(y0, y1)]
    gap = None
    seen = False
    limit = int(len(counts) * 0.62)
    valley = None
    for index, count in enumerate(counts[:limit]):
        if count >= 2:
            seen = True
        if not seen or index < 2:
            continue
        below = max(counts[index + 1 : index + 10], default=0)
        if count <= 1 and below >= 3:
            gap = index
            break
        if (
            valley is None
            and count <= 2
            and counts[index - 1] >= count + 2
            and below >= count + 4
        ):
            valley = index
    if gap is None:
        gap = valley
    if gap is None or gap < 2:
        return None
    mask = [list(rows[y0 + y][x0:x1]) for y in range(gap)]
    dots = _two_column_dots(mask)
    if dots:
        return dots
    blobs = _accent_blobs(mask)
    if not blobs:
        return None
    return _mark_from_blobs(blobs)


def _is_below_mark(mark: str) -> bool:
    if unicodedata.combining(mark) in {202, 220, 222}:
        return True
    return "BELOW" in unicodedata.name(mark, "")


def _with_printed_mark(char: str, mark: str) -> str:
    """Attach one combining mark. NFC turns it into the precomposed letter when one exists."""
    decomposed = unicodedata.normalize("NFD", char)
    base = decomposed[0]
    below = _is_below_mark(mark)
    kept = [item for item in decomposed[1:] if _is_below_mark(item) != below]
    return unicodedata.normalize("NFC", base + "".join(kept) + mark)


def _printed_below(rows: list[bytearray], box: tuple[int, int, int, int]) -> str | None:
    """Cedilla, ogonek, or dot below, when that ink is detached under the letter."""
    height, width = len(rows), len(rows[0])
    x0 = max(0, box[0])
    y0 = max(0, box[1])
    x1 = min(width, box[2])
    y1 = min(height, box[3])
    if y1 - y0 < 10 or x1 - x0 < 4:
        return None
    counts = [sum(rows[y][x0:x1]) for y in range(y0, y1)]
    start = int(len(counts) * 0.55)
    gap = None
    for index in range(start, len(counts) - 2):
        if counts[index] <= 1 and counts[index - 1] >= 3 and max(counts[index + 1 :], default=0) >= 3:
            gap = index
            break
    if gap is None:
        return None
    mask = [list(rows[y0 + y][x0:x1]) for y in range(gap + 1, len(counts))]
    blobs = _accent_blobs(mask)
    if not blobs:
        return None
    points = max(blobs, key=len)
    blob_w, blob_h = _blob_size(points)
    if _is_dot(points):
        return "\u0323"
    # A cedilla hangs down. An ogonek trails toward the right.
    xs = [point[0] for point in points]
    center = (min(xs) + max(xs)) / 2
    body_center = (x1 - x0) / 2
    if center > body_center + 1 and blob_w >= blob_h:
        return "\u0328"
    if blob_h >= 3:
        return "\u0327"
    return None


def _same_letter(host: str, read: str) -> bool:
    host_base, read_base = _skeleton(host), _skeleton(read)
    return host_base == read_base or {host_base, read_base} == {"f", "i"}


def _box_counts(rows: list[bytearray], box: tuple[int, int, int, int]) -> list[int] | None:
    height, width = len(rows), len(rows[0])
    x0 = max(0, box[0])
    y0 = max(0, box[1])
    x1 = min(width, box[2])
    y1 = min(height, box[3])
    if x1 - x0 < 4 or y1 - y0 < 8:
        return None
    return [sum(rows[y][x0:x1]) for y in range(y0, y1)]


def _is_strict_macron(rows: list[bytearray], box: tuple[int, int, int, int]) -> bool:
    """A flat bar at the top of the glyph, with empty rows under it."""
    counts = _box_counts(rows, box)
    if not counts or max(counts[0], counts[1] if len(counts) > 1 else 0) < 6:
        return False
    gap = None
    for index in range(2, min(len(counts), 8)):
        if counts[index] == 0 and counts[index - 1] == 0:
            gap = index - 1
            break
    if gap is None or gap > 4:
        return False
    height, width = len(rows), len(rows[0])
    x0 = max(0, box[0])
    y0 = max(0, box[1])
    x1 = min(width, box[2])
    mask = [list(rows[y0 + y][x0:x1]) for y in range(gap)]
    if _two_column_dots(mask):
        return False
    cols = [sum(row[x] for row in mask) for x in range(len(mask[0]))]
    ink = [index for index, count in enumerate(cols) if count]
    if len(ink) < 8 or ink[-1] - ink[0] != len(ink) - 1:
        return False
    return True


def _long_acute(rows: list[bytearray], box: tuple[int, int, int, int]) -> bool:
    """A steep stroke above a narrow i. A wide box has swallowed the next letter."""
    height, width = len(rows), len(rows[0])
    x0 = max(0, box[0])
    y0 = max(0, box[1])
    x1 = min(width, box[2])
    y1 = min(height, box[3])
    if x1 - x0 > 14 or y1 - y0 < 16:
        return False
    counts = _box_counts(rows, box)
    if not counts or max(counts[0], counts[1] if len(counts) > 1 else 0) < 2:
        return False
    gap = None
    for index in range(2, int(len(counts) * 0.55)):
        if index + 1 < len(counts) and counts[index] == 0 and counts[index + 1] == 0:
            gap = index
            break
    if gap is None:
        return False
    points = [
        (x, y)
        for y in range(y0, y0 + gap)
        for x in range(x0, x1)
        if rows[y][x]
    ]
    if len(points) < 16 or _slope(points) > -0.6:
        return False
    blob_w, blob_h = _blob_size(points)
    if blob_w < 5 or blob_h < 5:
        return False
    fill = len(points) / (blob_w * blob_h)
    return fill <= 0.75


def _is_inverted_question(rows: list[bytearray], box: tuple[int, int, int, int]) -> bool:
    """A dot over a hook. An i has a stem about as wide as its dot."""
    counts = _box_counts(rows, box)
    if not counts:
        return False
    gap = None
    for index in range(2, min(len(counts) // 2, 16)):
        if counts[index] <= 1 and counts[index + 1] <= 1 and max(counts[:index]) >= 2:
            gap = index
            break
    if gap is None:
        return False
    height, width = len(rows), len(rows[0])
    x0 = max(0, box[0])
    y0 = max(0, box[1])
    x1 = min(width, box[2])
    y1 = min(height, box[3])
    def span(y_start: int, y_end: int) -> int:
        ink = [x for y in range(y_start, y_end) for x in range(x0, x1) if rows[y][x]]
        return (max(ink) - min(ink) + 1) if ink else 0
    dot_w = span(y0, y0 + gap)
    hook_w = span(y0 + gap + 1, y1)
    mid = y0 + gap + max(1, (y1 - (y0 + gap)) // 2)
    middle_w = span(y0 + gap + 1, mid)
    bottom_w = span(mid, y1)
    # A hook flares toward the bottom. An i stem stays the same width.
    return (
        2 <= dot_w <= 6
        and hook_w >= 10
        and hook_w >= dot_w * 2.2
        and bottom_w >= middle_w + 3
    )


def _splice_apostrophe(host: str, boxed: str) -> str | None:
    """Put back an apostrophe Scan text folded into the next letter."""
    box = [char for char in boxed if char.isalpha() or char in "'’‘`"]
    if not any(char in "'’‘`" for char in box):
        return None
    host_chars = list(host)
    index = 0
    rebuilt: list[str] = []
    used = False
    for char in box:
        if char in "'’‘`":
            if index < len(host_chars) and host_chars[index] in "'’‘`":
                rebuilt.append(host_chars[index])
                index += 1
            elif index < len(host_chars) and host_chars[index] in "tl1I|":
                rebuilt.append("'")
                index += 1
            else:
                rebuilt.append("'")
            used = True
            continue
        while index < len(host_chars) and not host_chars[index].isalpha():
            rebuilt.append(host_chars[index])
            index += 1
        if index >= len(host_chars) or not _same_letter(host_chars[index], char):
            return None
        rebuilt.append(host_chars[index])
        index += 1
    while index < len(host_chars):
        rebuilt.append(host_chars[index])
        index += 1
    if not used:
        return None
    result = "".join(rebuilt)
    return result if result != host else None


def _fit_letters(
    host: list[str], letters: list[tuple[str, bool]]
) -> list[tuple[str, bool]] | None:
    """Match character boxes to a word, dropping one repeated letter beside its accented form."""
    def matches(trial: list[tuple[str, bool]]) -> bool:
        return len(trial) == len(host) and all(
            _same_letter(host_char, read_char) for host_char, (read_char, _changed) in zip(host, trial)
        )

    if matches(letters):
        return letters
    if len(letters) != len(host) + 1:
        return None
    for index in range(len(letters) - 1):
        left, right = letters[index][0], letters[index + 1][0]
        if not _same_letter(left, right):
            continue
        drop_left = _mark_count(right) >= _mark_count(left)
        trial = letters[:index] + letters[index + 1 :] if drop_left else letters[: index + 1] + letters[index + 2 :]
        if matches(trial):
            return trial
    return None


def _repair_printed_accents(image_path: Path, lines: list[Line], tessdata: Path) -> None:
    """Read accents, apostrophes, and ¿ from the glyph ink and the character boxes."""
    rows, height = _ink_rows(image_path)
    boxes = _char_boxes(image_path, tessdata, height)
    if not boxes:
        return
    corrected: list[tuple[str, int, int, int, int, bool]] = []
    for char, x0, y0, x1, y1 in boxes:
        box = (x0, y0, x1, y1)
        pieces = unicodedata.normalize("NFD", char)[1:] if char.isalpha() else ""
        has_above = any(not _is_below_mark(item) for item in pieces)
        has_below = any(_is_below_mark(item) for item in pieces)
        detected = _printed_mark(rows, box) if char.isalpha() else None
        replacement = char
        changed = False
        if char.isalpha() and _is_strict_macron(rows, box):
            replacement = _with_printed_mark(char, "\u0304")
            changed = True
        elif (
            has_above
            and detected is None
            and "\u0302" in pieces
        ):
            # A circumflex the shape reader did not contradict.
            replacement = char
            changed = True
        elif has_above and detected:
            replacement = _with_printed_mark(char, detected)
            changed = True
        elif has_below:
            below = _printed_below(rows, box)
            if below:
                replacement = _with_printed_mark(char, below)
                changed = True
        if (
            not changed
            and _skeleton(char) in {"s", "n", "r"}
            and _printed_below(rows, box) == "\u0323"
        ):
            replacement = _with_printed_mark(char, "\u0323")
            changed = True
        if not changed and _skeleton(char) == "i" and _long_acute(rows, box):
            replacement = _with_printed_mark(char, "\u0301")
            changed = True
        corrected.append((replacement, x0, y0, x1, y1, changed))
    for line in lines:
        for word in line.words:
            inside = [
                item
                for item in corrected
                if word.x - 3 <= (item[1] + item[3]) / 2 <= word.x + word.w + 3
                and word.y - 6 <= (item[2] + item[4]) / 2 <= word.y + word.h + 6
            ]
            inside.sort(key=lambda item: item[1])
            if inside and _is_inverted_question(rows, (inside[0][1], inside[0][2], inside[0][3], inside[0][4])):
                if word.text[:1] in "iIlZz":
                    word.text = "\u00bf" + word.text[1:]
            spliced = _splice_apostrophe(word.text, "".join(item[0] for item in inside))
            if spliced:
                word.text = spliced
            letters = [(item[0], item[5]) for item in inside if any(ch.isalpha() for ch in item[0])]
            host = [char for char in word.text if char.isalpha()]
            fitted = _fit_letters(host, letters)
            if fitted is None:
                continue
            letters = fitted
            rebuilt: list[str] = []
            index = 0
            for char in word.text:
                if not char.isalpha():
                    rebuilt.append(char)
                    continue
                replacement, changed = letters[index]
                index += 1
                host_marks = unicodedata.normalize("NFD", char)
                read_marks = unicodedata.normalize("NFD", replacement)
                # The tail of s at the end of a word is not a dot underneath it.
                if (
                    index == len(letters)
                    and "\u0323" in read_marks
                    and "\u0323" not in host_marks
                ):
                    rebuilt.append(char)
                    continue
                if changed:
                    rebuilt.append(replacement)
                elif (
                    char.isascii()
                    and replacement[:1].islower()
                    and _has_latin_extension(replacement)
                    and _same_letter(char, replacement)
                ):
                    rebuilt.append(replacement)
                elif _skeleton(char) == "f" and _skeleton(replacement) == "i":
                    rebuilt.append(replacement)
                else:
                    rebuilt.append(char)
            word.text = "".join(rebuilt)
    _insert_orphan_quotes(lines, boxes)


def _insert_orphan_quotes(lines: list[Line], boxes: list[tuple[str, int, int, int, int]]) -> None:
    """A quote Scan text dropped, placed on the line it sits beside."""
    quotes = {"\u201c", "\u201d", "\u2018", "\u2019", "\u00ab", "\u00bb"}
    placed = [word for line in lines for word in line.words]
    for char, x0, y0, x1, y1 in boxes:
        if char not in quotes:
            continue
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        if any(
            word.x - 4 <= center_x <= word.x + word.w + 4
            and word.y - 6 <= center_y <= word.y + word.h + 6
            for word in placed
        ):
            continue
        nearest = min(
            lines,
            key=lambda line: abs(min(word.y for word in line.words) - y0),
            default=None,
        )
        if nearest is None:
            continue
        word = Word(char, x0, y0, max(1, x1 - x0), max(1, y1 - y0))
        nearest.words.append(word)
        nearest.words.sort(key=lambda item: item.x)
        placed.append(word)


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
        if _accent_conflict(host.text, candidate.text):
            continue
        if same_word and _mark_count(candidate.text) >= _mark_count(host.text):
            host.text = candidate.text
        elif same_ink and _mark_count(candidate.text) > _mark_count(host.text):
            host.text = candidate.text


def _letter_marks(text: str) -> list[tuple[str, str]]:
    letters: list[tuple[str, str]] = []
    for char in unicodedata.normalize("NFD", text):
        if unicodedata.combining(char):
            if letters:
                base, marks = letters[-1]
                letters[-1] = (base, marks + char)
            continue
        if char.isalpha():
            letters.append((char.casefold(), ""))
    return letters


def _accent_conflict(host: str, candidate: str) -> bool:
    """Do not swap a mark that Scan text already read for a different mark."""
    host_letters = _letter_marks(host)
    candidate_letters = _letter_marks(candidate)
    if len(host_letters) != len(candidate_letters):
        return False
    for (host_base, host_marks), (candidate_base, candidate_marks) in zip(
        host_letters, candidate_letters
    ):
        if host_base != candidate_base:
            return False
        if host_marks and host_marks != candidate_marks:
            return True
    return False


def _merge_script_lines(lines: list[Line], extra: list[Word]) -> None:
    """Add Japanese, Chinese, Korean, Tamil, Thai, Arabic, Malayalam, and Telugu lines Scan text skipped.

    One false letter on an English page is ignored. A real script line has
    several characters, and that whole line is kept in reading order.
    """
    strong = [word for word in extra if word.conf >= 85 and _non_latin_script(word.text)]
    if sum(_script_letters(word.text) for word in strong) < 3:
        return
    for cluster in _cluster_words(strong):
        if sum(_script_letters(word.text) for word in cluster) < 3:
            continue
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
        if latin_letters >= 4:
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
    """Any Latin letter outside ASCII, including every accented form Unicode defines."""
    for ch in text:
        if unicodedata.category(ch) == "Mn":
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
    special = sum(1 for ch in letters if _script(ch) in _SCRIPTS)
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
    if 0x0E00 <= code <= 0x0E7F:
        return "thai"
    if (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    ):
        return "arabic"
    if 0x0D00 <= code <= 0x0D7F:
        return "malayalam"
    if 0x0C00 <= code <= 0x0C7F:
        return "telugu"
    return "other"


async def _ocr_lines(image_path: Path, engine) -> tuple[list[Line], str]:
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
    fallback = (result.text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return lines, fallback


async def recognize_page(image_path: Path, engine, tessdata: Path | None) -> tuple[str, list[PlacedLine]]:
    """Read one page. Returns the text and each line's box in image pixels."""
    lines, fallback = await _ocr_lines(image_path, engine)
    if not lines:
        return fallback, []
    text = apply_character_repairs(image_path, lines, tessdata)
    return text, placed_lines(lines)


async def recognize(image_path: Path, engine, tessdata: Path | None) -> str:
    """Read one page image with Scan text, then repair special characters."""
    text, _placed = await recognize_page(image_path, engine, tessdata)
    return text


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
