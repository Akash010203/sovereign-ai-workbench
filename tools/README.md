# tools/ (reserved)

Empty until **Phase 10** (see `docs/ARCHITECTURE.md` §11).

Planned contents: `registry.py` (controlled tool registry — every
agent tool call passes through this, nothing is executed directly),
`filesystem.py`, `calculator.py`, `pdf.py`, `ocr.py`,
`spreadsheet.py`, `word.py`, `powerpoint.py`, `code_sandbox.py`. Each
tool gets: name, description, allowed inputs, validation, error
handling, output schema, audit log entry.
