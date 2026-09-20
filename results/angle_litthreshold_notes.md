# Angle: Literature threshold for domain-invariant / few-shot / meta-learning success on small multi-site datasets

**Question:** Do published successes of IRM/DANN/MAML/TabPFN-style methods on small (n<300)
multi-site engineering/physical-science datasets define a sample-size or facility-count
threshold, and does our corpus (n=114, 4 facilities, 8–45 pts/facility, target = continuous
Cp in log space) sit above or below it?

**Method:** Web literature search only (no new code/data run — this is a literature angle).
Searched for (a) domain-generalization theory on minimum number of environments, (b) the
standard DG benchmark literature (DomainBed) as the field's own stress-test of DANN/IRM/
meta-learning at "few domains," and (c) individual published "small-N meta-learning/DANN
success" papers, inspected closely enough to see what "small" and "task" actually meant in
each case (real domain shift vs. synthetic task construction; true N vs. hidden effective N).

## Finding 1 — The field's own benchmark says domain count ≈ ours is not enough even with far more data per domain

Gulrajani & Lopez-Paz, *"In Search of Lost Domain Generalization"* (ICLR 2021,
arXiv:2007.01434) built DomainBed specifically to stress-test DG algorithms rigorously.
All 7 of its datasets use **at most 6 training domains** (PACS: 4 domains, ~1,670
images/domain; Office-Home: 4 domains, ~2,427 images/domain; DomainNet: 6 domains, ~600k
images total) — i.e. domain counts in the same range as our 4 facilities. Comparing 13
popular DG algorithms (including DANN, IRM, and meta-learning-based methods) under
identical protocol, they conclude: *"no competitor is able to outperform ERM by more than
one point"* — even with hundreds to thousands of samples per domain. This is the closest
thing the field has to a null result at our domain count, and it is a well-cited,
foundational finding, not a fringe result.

Zhang et al., *"Lost Domain Generalization Is a Natural Consequence of Lack of Training
Domains"* (AAAI), goes further: they attribute the DomainBed failure directly to too few
domains, and show OOD test accuracy increases **monotonically with number of training
domains** — i.e. the fix the literature itself proposes is "get more domains," not "use a
cleverer architecture." Our corpus structurally cannot get more than 4 without new facility
data.

## Finding 2 — Theory: IRM needs environment count to scale with confound dimensionality

Rosenfeld, Ravikumar & Risteski, *"The Risks of Invariant Risk Minimization"* (ICLR 2021,
arXiv:2010.05761): even in the simple linear case, IRM needs **the number of training
environments to exceed the dimensionality of the spurious/nuisance feature space** to
recover the true invariant predictor; in the nonlinear regime it can fail catastrophically
unless test data are already similar to train data (the exact thing it's meant to fix), and
then "does not fundamentally improve over ERM." Under strict LOFO we train on only 3
environments at a time. Given that facility identity here is entangled with several
geometry/measurement covariates (geometry_type, chamber vs. cylinder configuration, axial
vs. radial gap regime — see project GOTCHA about geom_confidence confound), it is very
plausible the spurious-feature dimensionality is ≥3, i.e. right at or past the point where
Rosenfeld's own toy linear example already breaks down. This is a plausibility argument, not
a proof for our exact data, but it is directly the right shape of argument, from the paper
that specifically characterizes IRM's minimum-environment requirement.

## Finding 3 — Inspecting actual "small-N meta-learning succeeded" engineering papers: none are a real analog

I checked the concrete examples that came up as apparent counter-evidence, reading past the
abstract into methodology:

- **GFRP tubular concrete columns, Reptile meta-learning** (PMC11236119, "small sample deep
  meta learning"): real n = **72 specimens total** (66 train / 6 test) — close to n<300 —
  but from **one single research group's single experimental campaign** (not multi-site),
  and the meta-learning "tasks" are **synthetic**: a genetic algorithm augments the 66 real
  points into **15,000 synthetic samples** used to train Reptile. There is no real domain
  shift being bridged at all — this is data-scarcity-via-augmentation on one homogeneous
  distribution, a fundamentally easier and different problem than cross-facility
  generalization. R²=0.982 reported, but it is not a fair analog to our task.
- **Concrete strength MAML+SHAP** (Springer AI in Civil Engineering, paywalled, abstract-level
  only): claims beat XGBoost/RF (MAE 3.56 MPa); could not verify task/domain construction
  behind the paywall, so it is not usable as a citable counter-example without that check —
  flagged as unverified rather than claimed.
- **Bearing / wind-turbine "few-shot" fault diagnosis** (CWRU, Paderborn, and multiple
  MAML/Reptile/prototypical-network papers): these are the most-cited "meta-learning works
  on small engineering data" results, but on inspection: (a) the task is **classification**
  (fault type), not continuous physical-law regression; (b) "K-shot" refers to K labeled
  examples **per class at meta-test time**, drawn from an underlying pool of **thousands of
  raw vibration signal segments** (12–64 kHz sampling, windowed into many instances per
  physical run) — the real underlying N is orders of magnitude larger than the "few-shot"
  framing suggests, and totally unlike our 8–45 independent scalar rows per facility.
- **Closest genuine match by raw headline N**: Chan et al. (PMC10079633), domain-adversarial
  learning (DANN) for oral-cancer diagnosis from two imaging centers, Doha (n=67) and Dallas
  (n=46) — **total n=113, almost exactly our n=114**, and only 2 domains (fewer than our 4,
  which is the easier case). DANN beat the best single-domain baseline by **+2.54 points
  average sensitivity/specificity (p=0.034)** — a genuine, statistically significant, but
  modest win, and the single most relevant precedent found. Critically, though, it is (a) a
  binary classification task, not continuous regression of a physical law, and (b) the model
  is not actually trained on 113 independent scalars: each image contributes a 160×160 grid
  of pixels with an 880-timepoint fluorescence-decay curve per pixel, giving **"over 2.5
  million pixels to train"** — a ~20,000× hidden multiplication of effective training
  resolution via spatial/spectral sub-sampling that has no analog for our scalar rotor-rig
  measurements (Re, Cp, geometry ratios) — there is no way to extract 2.5 million correlated
  sub-observations from a single (Re_Omega, Cp) pair.

## Conclusion (honest, matched to the project's own decision rule)

The literature does **not** show that n=114 across 4 facilities (8–45 pts/facility) is an
arbitrary or unusually harsh cutoff invented for this paper — it converges from three
independent directions on the same answer:

1. The field's own most rigorous benchmark (DomainBed) shows DANN/IRM/meta-learning provide
   **no reliable edge over plain pooled regression even with a comparable domain count and
   100–1000× more samples per domain** than we have.
2. IRM-family theory (Rosenfeld et al.) predicts these methods provably need environment
   counts that scale with confound dimensionality — 3 usable environments under LOFO is
   very plausibly already below that bar here.
3. Every published "success" story close to our N either (a) is not actually a cross-domain
   problem (single source + synthetic task augmentation), or (b) is classification with a
   hidden per-instance sample multiplication of 3–4 orders of magnitude (raw signal windows,
   or millions of pixels) that a scalar physical measurement cannot replicate.

**Verdict for this angle: this is a genuine, citable, non-cherry-picked negative-but-informative
finding.** It does not produce a new working predictive model (so it does not clear the
project's "pooled R² positive AND every held-out facility R²≥0" bar, nor should it try to —
that bar is for empirical model claims, not literature framing). What it DOES honestly
provide is a well-sourced answer to "is this corpus just too small, or did we not try hard
enough": **the corpus is below every threshold at which comparable methods have any
published, mechanism-matched success**, and the specific missing ingredient in every
close-by "success" (many more domains, or a hidden 10³–10⁴× per-instance sample
multiplication, or no real domain shift at all) is not something more clever modeling can
manufacture from 114 real scalar physical measurements. This is a legitimate,
non-overclaimed contribution to add as literature-grounded discussion/context, explaining
*why* the 16 empirical method failures were expected rather than a modeling failure on the
authors' part — but it should be framed exactly that way (an explanatory/diagnostic
contribution) and not spun as a new positive experimental result.

## Sources
- Gulrajani, I. & Lopez-Paz, D. "In Search of Lost Domain Generalization." ICLR 2021.
  arXiv:2007.01434. https://arxiv.org/pdf/2007.01434
- Zhang, Y. et al. "Lost Domain Generalization Is a Natural Consequence of Lack of Training
  Domains." AAAI. https://hongyanz.github.io/publications/AAAI_Lost.pdf
- Rosenfeld, E., Ravikumar, P., Risteski, A. "The Risks of Invariant Risk Minimization."
  ICLR 2021. arXiv:2010.05761. https://arxiv.org/abs/2010.05761
- "Data modeling analysis of GFRP tubular filled concrete column based on small sample deep
  meta learning method." PMC11236119. https://pmc.ncbi.nlm.nih.gov/articles/PMC11236119/
- "Few-shot meta-learning for concrete strength prediction: a model-agnostic approach with
  SHAP analysis." AI in Civil Engineering (Springer), 2025.
  https://link.springer.com/article/10.1007/s43503-025-00064-8 (abstract-level only,
  paywalled — flagged unverified, not used as evidence for or against)
- Chan, K.C. et al. "Aligning Small Datasets Using Domain Adversarial Learning: Applications
  in Automated in Vivo Oral Cancer Diagnosis." PMC10079633.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10079633/
- Hollmann, N. et al. "TabPFN: A Transformer That Solves Small Tabular Classification
  Problems in a Second." arXiv:2207.01848. https://arxiv.org/pdf/2207.01848 (confirms
  TabPFN's validated regime is i.i.d. small-N classification up to ~1000 samples, NOT
  cross-domain/OOD generalization — consistent with this project's own empirical LOFO
  failure of TabPFN)
- Background/context on DANN training instability with few source domains (multiple
  secondary sources via search; general mechanism, not a single primary citation).
