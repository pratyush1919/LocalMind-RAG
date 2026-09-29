import re
import logging
from dataclasses import dataclass

from llama_index.core import Document

logger = logging.getLogger(__name__)


from dataclasses import dataclass

@dataclass
class Section:
    title: str
    level: int
    content: str

    heading_level_1: str | None = None
    heading_level_2: str | None = None
    heading_level_3: str | None = None

HEADING_PATTERNS = [
    r"^chapter\s+\d+",
    r"^part\s+[ivxlcdm]+",
    r"^\d+\.\d+$",
    r"^\d+\.\d+\.\d+$",
    r"^\d+\.\d+\s+",
    r"^#+\s+",
]


def is_heading(text: str) -> bool:
    text = text.strip().lower()

    if not text:
        return False

    for pattern in HEADING_PATTERNS:
        if re.match(pattern, text):
            return True

    return False

def get_heading_level(text: str):

    text = text.strip().lower()

    if re.match(r"^chapter\s+\d+", text):
        return 1

    if re.match(r"^part\s+[ivxlcdm]+", text):
        return 1

    if re.match(r"^\d+\.\d+\b", text):
        return 2

    if re.match(r"^\d+\.\d+\.\d+\b", text):
        return 3

    return None

def split_into_sections(text: str):

    lines = text.splitlines()

    sections = []

    current_title = "Introduction"
    current_content = []

    current_h1 = None
    current_h2 = None
    current_h3 = None

    for line in lines:

        if is_heading(line):

            if current_content:

                sections.append(
                    Section(
                        title=current_title,
                        level=1,
                        content="\n".join(current_content),
                        heading_level_1=current_h1,
                        heading_level_2=current_h2,
                        heading_level_3=current_h3,
                    )
                )

            level = get_heading_level(line)

            if level == 1:
                current_h1 = line.strip()
                current_h2 = None
                current_h3 = None

            elif level == 2:
                current_h2 = line.strip()
                current_h3 = None

            elif level == 3:
                current_h3 = line.strip()

            current_title = line.strip()
            current_content = []

        else:
            current_content.append(line)

    if current_content:

        sections.append(
            Section(
                title=current_title,
                level=1,
                content="\n".join(current_content),
                heading_level_1=current_h1,
                heading_level_2=current_h2,
                heading_level_3=current_h3,
            )
        )

    return sections

def is_navigation_section(section: Section):
    """
    Detect table-of-contents / navigation-only sections.

    Generic heuristic.
    """

    text = section.content.lower()

    heading_count = len(
        re.findall(
            r"(chapter|part|\d+\.\d+)",
            text,
        )
    )

    word_count = len(text.split())

    if heading_count >= 5 and word_count < 300:
        return True

    return False


def split_document_into_section_docs(doc):
    """
    Convert one large Document into
    multiple section-aware Documents.
    """

    sections = split_into_sections(doc.text)

    section_docs = []

    for section in sections:

        if is_navigation_section(section):

            logger.info(
                f"[Structure] Skipping navigation section: "
                f"{section.title}"
            )

            continue

        section_docs.append(
            Document(
                text=f"{section.title}\n\n{section.content}",
        metadata={
            **doc.metadata,
            "heading_level_1": section.heading_level_1,
            "heading_level_2": section.heading_level_2,
            "heading_level_3": section.heading_level_3,
            "chunk_type": "content",
        },
            )
        )

    logger.info(
        f"[Structure] {len(sections)} sections -> "
        f"{len(section_docs)} content sections"
    )

    for d in section_docs[:20]:

        print("\n----------------")    
        print("H1:", d.metadata.get("heading_level_1"))
        print("H2:", d.metadata.get("heading_level_2"))
        print("H3:", d.metadata.get("heading_level_3"))
    return section_docs