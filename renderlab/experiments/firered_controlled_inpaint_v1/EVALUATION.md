# FireRed controlled inpainting: completed case

Job `1bdff9f9-9fdf-4094-94b2-a08c6dfd8baa` is the only inference run.

| Criterion | Raw FireRed | Deterministic production composite |
|---|---|---|
| Outside-mask semantics | PASS, 5/5 | Original source pixels preserved |
| Exact outside-mask pixels | FAIL: 779,652 changed (76.2141%), max error 27/255 | PASS: 0 changed, max error 0 |
| Masked instruction | PASS, 4/5; 90.09399% blue coverage, somewhat flat texture | Identical raw pixels inside mask |
| Boundary | 4/5: minor thin rim, no severe tearing/spill | 4/5: inner rim retained, outer pixels restored exactly |

The frozen raw protocol remains **FAIL**. Full-frame VAE drift is consistent with
the distributed small differences; an isolated VAE-only control was not run.
The production composite is a supplementary deterministic containment result,
not evidence that raw FireRed guarantees exact pixel preservation.

Both images are RGB 1160×896. No alpha or color-mode conversion was performed.
The native CPU ImageCompositeMasked result was saved, decoded, and checked
against exact selection of source pixels outside and raw pixels inside the mask.
The raw PNG's SHA-256 remains unchanged. No new uploads or inference occurred.
