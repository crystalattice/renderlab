Final disposition: **FAIL** for a controlled face-identity swap.

Evaluated the original target, identity reference and unchanged raw output locally.
Job: `f1236b3d-144f-49d4-bc68-99cc034432fd`, seed 3407. No Cloud access or generation
was used for evaluation. Scores are subjective visual judgments on RenderLab's
1–5 scale: higher is better except artifact severity, where lower is better.
The experiment supplies thresholds but no separate verbal anchors for each number.

| Existing criterion | Score | Required | Result |
|---|---:|---:|---|
| Identity transfer (`identity_preservation`) | 2 | ≥4 | Fail |
| Body/morphology preservation | 5 | 5 | Pass |
| Pose preservation | 5 | 5 | Pass |
| Scene preservation | 4 | ≥4 | Pass |
| Face-boundary quality | 4 | ≥4 | Pass |
| Instruction adherence | 2 | ≥4 | Fail |
| Artifact severity | 2 | ≤2 | Pass |

The output retains the target-like face outline, forehead mark, nose and mouth.
The reference's eye/brow presentation, cheek/jaw appearance and fuller-looking lips
are not convincingly transferred. Hair color is not evidence of facial identity
transfer. This is a visual resemblance judgment, not biometric identification.

The seated lean, raised arms, bent elbows, handle grips, leg placement and head
orientation are preserved. Visible torso and limb proportions show no clear
reshaping. These scores concern visible morphology and pose, not exact pixels or
anatomy hidden by clothing. The planned resize/crop from 1160 × 896 to 1184 × 880
is accounted for rather than treated as an unexpected edit.

Clothing preservation scores **4/5**: the opaque white T-shirt, sleeves, long hem
and broad fold pattern remain, with some minor redraw. The gym's red pads, black
machine uprights, cables, weights and dark background remain. The reference's
jacket and rainy street have not replaced the target clothing or environment.

Hairstyle and visible hair-length preservation both score **1/5 — failure**.
The target's brown hair is swept back/tied behind the head, with short loose front
strands. The output introduces dark, glossy, loose shoulder-length waves. The
target's true untied length cannot be inferred, but its visible silhouette and
length clearly changed. This violates the preservation requirement even if the
new hair resembles the reference.

Face/neck/skin-transition quality scores **4/5**: the jaw-to-neck edge is continuous
and skin tone broadly fits the warm flash-lit target. There is no conspicuous
halo, doubled jaw or pasted boundary. Smoothing reduces fine facial texture, and
the new hair obscures portions of the side boundary.

Overall realism scores **4/5**, with artifact severity **2/5**. The result is a
plausible gym photograph at scene scale, but skin looks smoothed and dark hair
forms glossy, somewhat synthetic clumps. No obvious new extra digits or doubled
facial features are visible. The reflective/damaged-looking seat tip and strong
flash appearance are present in the target and are not charged as new defects.
Realism and preserved body/pose do not compensate for the identity and hair failures.

`evaluation.json` contains all scores, thresholds, observations and source/output
hashes. The raw PNG remains byte-for-byte unchanged, SHA-256:
`0be2d30b68b77415c16330f711c89b49984dc9df5d0603eb2e65553034fe5095`.

The labeled [local contact sheet](contact_sheet.local.png) includes full views and
face/hair crops. It contains source pixels and is intentionally ignored by Git;
neither source image is committed. Recreate it with `python3
renderlab/experiments/qwen_face_swap_clothed_v1/make_contact_sheet.py` on a machine
with the manifest inputs. Originals and raw output are only read by this script.

Validation: all **86 RenderLab tests passed**; the local experiment validator
passed **43 editor links, 15 API nodes, 21 API links, 42 inputs and six rejected
invalid mutations**. Evaluation threshold consistency, all three asset hashes,
and the Git exclusion of the contact sheet were also checked. Tests establish
record/graph integrity; visual quality is assessed above.
