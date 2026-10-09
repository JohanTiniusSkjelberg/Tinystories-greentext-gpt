"""Format data/greentext_darwin.txt: one greentext line per text line.

The raw file has one story per line, with JSON escapes (\\n, \\", \\u2019)
as literal text, and many '>' lines glued together without a line break.
Duplicate stories and stories with OCR noise are removed.
"""
import re
from collections import Counter
import sys
from pathlib import Path

EOT = "<|endoftext|>"
DATA = Path(__file__).parent / "data"

# A line that ends with one of these words was cut off by the OCR.
DANGLING = (
    "a an the to my and but or of for with from that their our your is was were are "
    "i he she we they so because as into which while if these those some has have had "
    "be been not than just very its im i'm it's he's she's they're can will would could "
    "should until every"
).split()

# Each rule finds one type of flaw in a line (without the '>' at the start).
LINE_FLAWS = {
    "noise character": re.compile(r"[\\|^`{}§»¢¥™=]|~(?!\d)|<(?!3)|>|#(?!\d)"),
    "lone underscore or dash": re.compile(r"(?<![A-Za-z])_|_(?![A-Za-z])|—$|—-|-—"),
    "unbalanced bracket": re.compile(r"^[^\[\]]*[\[\]][^\[\]]*$|^[^()]*\)|\([^()]*$"),
    "junk at line end": re.compile(r" (?:[A-HJ-Zb-hj-z]|[^\w\s\"'”’)\]!?.*%$€]{1,3})$|:$"),
    "junk word": re.compile(r"\b(?=\w*\d)(?=\w*[A-Z])(?=\w*[a-z])\w{5,}\b"),
    "cut-off line": re.compile(r" (?:%s)$" % "|".join(DANGLING), re.IGNORECASE),
}
SMILEY = re.compile(r"[:;]-?[()]")
MIN_LINES = 3

ESCAPES = {"n": "\n", "t": "\t", "r": "", '"': '"', "\\": "\\", "/": "/"}


def unescape(text):
    def repl(m):
        if m.group(1):
            return chr(int(m.group(1), 16))
        return ESCAPES.get(m.group(2), m.group(0))

    return re.sub(r"\\(?:u([0-9a-fA-F]{4})|(.))", repl, text)


def format_story(story):
    story = unescape(story)
    # Start a new line at each '>' that is glued to the text before it.
    # '<>', '->' and '=>' are not line starts.
    story = re.sub(r"(?<=[^\n<=-])[ \t]*>", "\n>", story)
    lines = [re.sub(r"\s+", " ", line).strip() for line in story.split("\n")]
    # No space after the '>' at the start of a line.
    lines = [re.sub(r"^>\s+", ">", line) for line in lines]
    return "\n".join(line for line in lines if line and line != ">")


def find_flaw(story):
    """Return the first flaw in the story, or None if the story is good."""
    lines = story.split("\n")
    if len(lines) < MIN_LINES:
        return "too short"
    for line in lines:
        if not line.startswith(">"):
            return "line without '>'"
        text = SMILEY.sub("", line[1:])
        for name, pattern in LINE_FLAWS.items():
            if pattern.search(text):
                return name
    return None


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "greentext_darwin.txt"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_clean.txt")

    stories = [format_story(s) for s in src.read_text(encoding="utf-8").split(EOT)]
    stories = [s for s in stories if s]
    total = len(stories)

    stories = list(dict.fromkeys(stories))
    removed = Counter({"duplicate": total - len(stories)})
    flaws = [find_flaw(s) for s in stories]
    removed.update(f for f in flaws if f)
    stories = [s for s, f in zip(stories, flaws) if not f]

    dst.write_text("".join(f"{s}\n{EOT}\n" for s in stories), encoding="utf-8")
    for name, count in removed.most_common():
        print(f"removed {count:4d}  {name}")
    print(f"{len(stories)} of {total} stories -> {dst}")


if __name__ == "__main__":
    main()
