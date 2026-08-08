# Object-Dependent Concept Brittleness in Diffusion Models

## Visual overview

### Prototype-guided correction

![Qualitative comparison between original Flux.1-dev generations and corrected results](assets/qualitative_correction.png)

Representative failures span styles and attributes such as ink sketch, oil
painting, vintage photo, color, material and 3D rendering. The corrected
outputs express the requested concept more clearly while preserving the main
object semantics.

### Intervention and reference-selection behavior

![Style correction under different intervention coefficients](assets/alpha_sensitivity_style.png)

Increasing the intervention coefficient $\alpha$ strengthens the requested
style, but overly large values can change layout or introduce semantic drift.
The appendix examples show the best repair--preservation balance around
$\alpha=0.8$.

![Reference-sample filtering under different thresholds](assets/reference_filtering_tau.png)

The top rows contain visually obvious failures. Lowering the filtering
threshold $\tau$ rejects more of them and yields purer class prototypes, at the
cost of retaining fewer and less diverse reference samples.

### Step-wise SAE diagnosis

<p align="center">
  <a href="assets/sae_trajectory_success.png"><img src="assets/sae_trajectory_success.png" width="49%" alt="Successful sample SAE trajectory"></a>
  <a href="assets/sae_trajectory_failure.png"><img src="assets/sae_trajectory_failure.png" width="49%" alt="Failed sample SAE trajectory"></a>
</p>

The successful *dog, ink sketch* sample (left) stays close to the class-mean
sparse code over the first ten denoising steps. The failed *orange, ink sketch*
sample (right) exhibits dominant-dimension mismatches from the early steps and
throughout the guided window. Click either panel to inspect it at full size.

## Model setup

Model weights are not distributed with the code. Put trusted local weights in
the relative layout documented in `models/README.md`, or override the model
path arguments shown by:

```bash
python generate.py --help
```

## Paper-aligned workflow

1. Extract classifier-free-guided denoising step updates:

   ```bash
   python generate.py --model sdxl --mode save_feature
   ```

2. Train one SAE per denoising timestep:

   ```bash
   python train_sae.py --data feature/step_updates_style_2026.pt \
     --d_model 16 --n_dirs 64 --checkpoint_dir checkpoints/sdxl
   ```

3. Select reliable reference samples in SAE space:

   ```bash
   python select_reference_samples.py \
     --data feature/step_updates_style_2026.pt \
     --checkpoint_dir checkpoints/sdxl --steps 0 1 2 \
     --remove_far --filtered_output feature/step_updates_style_filtered.pt
   ```

4. Build class-level concept prototypes:

   ```bash
   python build_concept_prototypes.py \
     --checkpoint_dir checkpoints/sdxl \
     --output_path feature/sdxl_style_prototypes.pt
   ```

5. Apply prototype-guided correction in the early denoising window:

   ```bash
   python generate.py --model sdxl --mode sae_style_guide \
     --ablation_steps 1,2,3,4,5,6,7,8,9,10
   ```
