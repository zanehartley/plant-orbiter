# PlantDreamer hybrid masks

Binary masks (255 = plant + pot) for PlantDreamer's real photos. Both halves of the paper use them:

- single-photo reconstruction: the input photo of each PlantDreamer plant is cut out with its mask, and every
  scored real view is compared against its mask (`reconstruction/eval_real_views.py`);
- text-to-3D: the real reference photos for the kernel distance (KD) are cut out with them
  (`evaluation/prep_reference.py`).

`plantdreamer_hybrid/<plant>/<view>.png` has the same size and file name as PlantDreamer's
`images/undistorted/<view>.png`. There are 4,045 masks over 20 plants (`counts.txt`). bean_2 has none, because the last
quarter or so of its dataset masks outline a different plant, so it is excluded from every experiment.

Each mask is hybrid = (F & D) | (F & dilate(D, 25 px) & upper 60% of the plant's height) (`make_hybrid_masks.py`):

- D is PlantDreamer's own mask (`images/mask/<view>.png.png`), which is loose around leaves.
- F is a rembg `isnet-general-use` cut-out of the raw photo.
- F sets the edges, D keeps the cloth and turntable out, and the second term restores the thin stems that D cuts off.

A quality gate (`qa_hybrid_masks.py`) then rejected a mask if it kept less than 60% of D's pot region (its bottom
quarter) or less than 60% of D's area. Those 144 views are listed in `rejected_views.txt`; they are neither scored nor
used as KD reference photos, and D is never used alone. Views with a missing or wrongly sized dataset mask (mint_4)
have no hybrid mask either.

The two scripts document how the masks were made and are not needed to reproduce the paper. rembg may not give
bit-identical masks on other hardware or versions, so compare per-plant counts with `counts.txt` when remaking them.
The masks are derived from the PlantDreamer dataset and are shared on the same terms as that dataset.
