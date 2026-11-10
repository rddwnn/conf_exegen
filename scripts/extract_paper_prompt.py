"""Extract and verify the Appendix B.2 ICL+CoT template using Poppler."""

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from common import sha256, write_json
    from downloads import PAPER_URL, download

    paper = ROOT / "data/references/exegen.pdf"
    download(PAPER_URL, paper)
    text = subprocess.check_output(["pdftotext", "-layout", str(paper), "-"], text=True)
    pages = text.split("\f")
    first = pages[23]
    start = first.index("ICL+CoT") + len("ICL+CoT")
    section = first[start:] + "\n" + pages[24].split("B.3")[0]
    lines = [
        line.strip()
        for line in section.splitlines()
        if line.strip() and line.strip() not in ("24", "25")
    ]
    # Undo PDF column wrapping only. Preserve wording, examples, and Unicode punctuation.
    prompt = "\n".join(lines)
    prompt = re.sub(r"(\w)-\n(\w)", r"\1\2", prompt)
    if "{student_history}" not in prompt or not prompt.endswith("Concept: Knowledge_Concept"):
        raise ValueError("Unexpected paper layout; verify Appendix B.2 manually")
    target = ROOT / "prompts/icl_cot_paper.txt"
    target.write_text(prompt + "\n", encoding="utf-8")
    write_json(
        ROOT / "prompts/provenance.json",
        {
            "icl_cot_paper.txt": {
                "source": PAPER_URL,
                "pdf_sha256": sha256(paper),
                "pages": [24, 25],
                "normalization": "strip indentation, blank lines, page numbers; join hyphenated wraps",
                "sha256": sha256(target),
            },
            "j_output.txt": {
                "status": "reconstructed: upstream txtfile/j_output.txt is unpublished"
            },
            "n_output.txt": {
                "status": "reconstructed: upstream txtfile/n_output.txt is unpublished"
            },
        },
    )
    print("Extracted original ICL+CoT template (layout normalization only)")


if __name__ == "__main__":
    main()
