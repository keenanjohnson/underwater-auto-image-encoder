## Fix grid line artifacts in tiled image processing

### Problem

Users on Windows reported visible grid line artifacts appearing in output images when processing large images through the GUI. These artifacts appeared as faint lines at regular intervals corresponding to tile boundaries.

### Root Cause

The tiled processing approach was processing each tile independently with reflection padding at the edges. When the model processed a tile, pixels near the boundaries were influenced by reflected content rather than actual neighboring image pixels. Even though adjacent tiles were blended together, they produced **different predictions** for the same pixel locations because they "saw" different context (reflected vs actual pixels).

### Solution

Added a `margin` parameter (default 64 pixels) to `process_image_tiled()` that extends each tile's processing region beyond its actual boundaries:

1. Extract an extended tile that includes extra margin pixels on all sides
2. Process the full extended region through the model
3. Crop away the margins, keeping only the center portion
4. Blend the cropped tiles as before

This ensures that when adjacent tiles overlap, they produce **identical predictions** for the overlapping region because the model had consistent context (actual neighboring pixels) during processing.

```
Before (artifacts):                    After (seamless):
┌─────────┐ ┌─────────┐               ┌───────────────┐
│  Tile 1 │ │  Tile 2 │               │  ┌─────────┐  │
└─────────┘ └─────────┘               │  │  Tile 1 │  │  ← Model sees neighbors
     ↑ reflection padding             │  └─────────┘  │
     causes different output          └───────────────┘
                                              ↓ crop
                                        ┌─────────┐
                                        │  Tile 1 │  ← Consistent predictions
                                        └─────────┘
```

### Changes

- `inference/inference.py`: Modified `process_image_tiled()` to extract extended tiles with margins, process them, then crop to the desired region
- Margin is automatically reduced for U-Shape Transformer models (which use smaller tiles)

### Testing

- Verified no syntax errors
- The fix applies to both U-Net and U-Shape Transformer models
- No changes to the public API (margin parameter has a sensible default)
