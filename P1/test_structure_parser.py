from P1.document_loader import load_document
from P1.structure_parser import split_document_into_section_docs

docs = load_document("data/Python.docx")

section_docs = []

for doc in docs:
    section_docs.extend(
        split_document_into_section_docs(doc)
    )

print(f"\nSections: {len(section_docs)}\n")

for d in section_docs[:30]:

    print("=" * 80)

    print("SECTION:")
    print(d.metadata.get("section_title"))

    print()

    print("H1:")
    print(d.metadata.get("heading_level_1"))

    print()

    print("H2:")
    print(d.metadata.get("heading_level_2"))

    print()

    print("H3:")
    print(d.metadata.get("heading_level_3"))

    print()

    print("WORDS:")
    print(len(d.text.split()))