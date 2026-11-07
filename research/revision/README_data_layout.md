# Revision experiments (official COMA split)

Everything for the post-3DV revision lives here. The old `data/` artefacts are untouched.

| Path | Contents |
|---|---|
| `data/` | dataset root: `processed_completion/`, `template.ply`, `norm_completion.pt`, `completion_test_samples/` (20 val samples for visual checks) |
| `checkpoints/` | trained weights, one subfolder per model/arm |
| `results/` | evaluation outputs (CSVs, completed meshes, figures) |
| `logs/` | console logs |

- Config: `research/configs/config_diffusion_revision.cfg`
- Raw train meshes: `data/train` (official COMA split, 18,978 meshes; the 1,483 test meshes were moved to `data/_revision/removed_from_train/`)
- **Test set:** `data/COMA_test_subset/gt` (1,483 meshes, 12 subjects, frames x91–x00 of every 100)
- Processing: `research/revision/process_official_split.py` (seed 1) → train 15,182 / val 3,796 (val = checkpoint selection only)
