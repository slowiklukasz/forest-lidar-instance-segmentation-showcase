# 🌲 3D Forest Instance & Semantic Segmentation from Synthetic LiDAR

**Transformer-based instance segmentation of individual trees (trunk + crown) from 3D point clouds — trained entirely on synthetic data, tested zero-shot on real forest scans.**

<p align="center">
  <img src="images\main.png" width="100%" alt="AI Dataset 1">
</p>

> Portfolio showcase — training/inference code is private. Happy to walk through it live on request, see [Contact](#contact).

---

## TL;DR

- Trained **100% on synthetic** forest scenes (own Blender-based generator, separate project) — no real annotated data used.
- Tested zero-shot on **[FOR-instance v2](https://paperswithcode.com/dataset/for-instance)**, a real, public forest LiDAR benchmark (ALS/UAV/MLS/TLS).
- Two-stage system: swappable backbone (SpConv / PointNet++ / PointNeXt / Point Transformer V3) + DETR-style transformer decoder with trunk-anchored queries and geometry-aware attention.
- Works at a budget-friendly **~100 pts/m² - 200 pts/m²** density, on free/low-tier cloud GPUs (Paperspace).
- **Works well** on simple, well-separated stands (spruce, pine, clean broadleaf). **Struggles** on dense, multi-story stands with heavy understory — that's the current focus.
- Reference point: [ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D) (SmartForest-no) — architecture diverges substantially from it.

---

## Synthetic training data

Procedurally generated forest scenes — trees, terrain, understory placed with collision-aware 3D logic, fully annotated (semantic + per-tree instance ID) at generation time, no manual labeling.

<br>
<p align="center">
   <i>Synthetic dataset example</i>
   <br><br>
  <img src="images\gif_3d_clear.gif" width="60%" alt="Synthetic dataset">
</p>

<p align="center">
   <i>Custom augmentation pipeline (20+ forest-specific transforms)</i>
   <br><br>
  <img src="images\aug_1.png" width="80%" alt="Augmentation examples">
  <img src="images\aug_4.png" width="80%" alt="Augmentation examples">
  <img src="images\aug_2.png" width="80%" alt="Augmentation examples">
  <img src="images\aug_3.png" width="80%" alt="Augmentation examples">
</p>

---

## Results

Simple conifer stands (spruce, pine) and well-separated broadleaf: strong results. Dense, multi-story stands with rich understory: still a weak point, and the main direction of ongoing work.

<br>
<p align="center">
   <i>Full-scene predictions on real, unseen forest plots (FOR-instance v2, ~100 pts/m²)</i>
</p>
<br>

<table align="center">
  <tr>
    <td><img src="images/NIBIO_plot_22.gif" width="100%" alt="Instance segmentation results"></td>
    <td><img src="images/CULS_plot_2.gif" width="100%" alt="Instance segmentation results"></td>
  </tr>
  <tr>
    <td><img src="images/NIBIO_MLS_MLS_burumPlot_2.gif" width="100%" alt="Instance segmentation results"></td>
    <td><img src="images/NIBIO_plot_3.gif" width="100%" alt="Instance segmentation results"></td>
  </tr>
  <tr>
    <td colspan="2" align="center"><img src="images/NIBIO_plot_6.gif" width="50%" alt="Instance segmentation results"></td>
  </tr>
</table>

<br>
<p align="center">
   <i>Inside a training step: GT vs. predicted semantics vs. query seeds vs. final instances</i>
   <br><br>
  <img src="images\res_3.png" width="90%" alt="AI Predictions">
</p>

<p align="center">
   <i>Full-scene inference — merged point clouds, predicted semantics and instances</i>
   <br><br>
  <img src="images\CULS_CULS_plot_2_annotated_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\NIBIO_NIBIO_plot_5_annotated_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\NIBIO_NIBIO_plot_17_annotated_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\NIBIO2_NIBIO2_plot27_annotated_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\SCION_SCION_plot_31_annotated_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\SCION_SCION_plot_35_annotated_val_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\RMIT_RMIT_test_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\NIBIO2_NIBIO2_plot47_annotated_val_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\CULS_CULS_plot_3_annotated_val_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\NIBIO_NIBIO_plot_23_annotated_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\NIBIO2_NIBIO2_plot3_annotated_test_merged_vis_pred.png" width="90%" alt="AI Predictions">
  <img src="images\NIBIO2_NIBIO2_plot54_annotated_val_merged_vis_pred.png" width="90%" alt="AI Predictions">
</p>

<p align="center">
  <img src="images/architecture_diagram.svg" width="80%" alt="Architecture diagram">
</p>

---

## Tech stack

`PyTorch` · `PyTorch Geometric` · `spconv` · custom Point Transformer V3 integration · `torchmetrics` · `scipy` · `laspy` · `Weights & Biases`

---

## Status

Active work in progress — architecture, augmentations, and evaluation are still being iterated on, especially sim-to-real transfer on dense/multi-story scenes. Happy to go deeper into any part of it in conversation.

## Contact

**Łukasz Słowik** — open to opportunities in 3D perception / point cloud ML / applied CV.
[[LinkedIn](https://www.linkedin.com/in/%C5%82ukasz-s%C5%82owik-650290226/)] · Email: slowik.lukasz1988@gmail.com