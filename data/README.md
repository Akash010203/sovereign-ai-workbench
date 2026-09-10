# Data directory

| folder | purpose |
|---|---|
| `raw/` | Original text exactly as obtained (or authored), untouched. |
| `cleaned/` | Whitespace-normalized, empty lines removed. |
| `processed/` | Length-filtered, split into `train.txt` / `val.txt`, ready for tokenization. |
| `metadata/` | One JSON file per raw source describing its origin, license, and processing notes. |
| `demo/` | Small files used specifically for the SIH live demo (Phase 23). Empty until then. |

Run `python scripts/prepare_data.py` to regenerate `cleaned/`,
`processed/`, and `metadata/` from whatever `.txt` files are in `raw/`.

Nothing is downloaded automatically. Every file placed in `raw/` must
have a corresponding entry recorded in `metadata/` (this is done
automatically by the pipeline script, but the policy is: **no silent
downloads, ever**).
