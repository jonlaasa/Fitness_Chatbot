from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def save_conversation(
    conversation_type: str,
    question: str,
    answer: str,
    retrieved_documents: list[dict],
    output_dir: str | Path,
    extra: dict | None = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(output_dir) / f"{conversation_type}_{timestamp}"
    base_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "conversation_type": conversation_type,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "question": question,
        "answer": answer,
        "retrieved_documents": retrieved_documents,
        "extra": extra or {},
    }

    (base_dir / "conversation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    txt_lines = [
        f"Type: {conversation_type}",
        f"Question: {question}",
        "",
        "Retrieved chunks:",
    ]
    for index, doc in enumerate(retrieved_documents, start=1):
        txt_lines.append(f"Chunk {index}:")
        txt_lines.append(f"- Parent document: {doc.get('title', 'unknown')}")
        txt_lines.append(f"- Source: {doc.get('source', 'unknown')}")
        metadata = doc.get("metadata")
        if metadata:
            txt_lines.append(f"- Metadata: {json.dumps(metadata, ensure_ascii=False)}")
        preview = doc.get("preview")
        if preview:
            txt_lines.append(f"- Chunk preview: {preview}")
        txt_lines.append("")
    txt_lines.extend(
        [
            "Answer:",
            answer,
            "",
        ]
    )
    (base_dir / "conversation.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    return base_dir
