# Frozen Pillar-0: Radiology Task Head Pipeline

Converts chest CT volumes into Pillar-0 embedding vectors, then trains and
compares classifiers on top of those vectors.

Two data sources are supported. Both produce the same output format, so the
second stage reads either one without modification.

This was the first stage of an AI platform for radiologists, now being developed
with Dr. Philip Edgcumbe and Armin Saghafian, continuing earlier work by
Dr. Sara Hosseinirad.

## What this does

A chest CT is roughly 512 x 512 x 300 voxels, about 78 million numbers.
Pillar-0, an open foundation model trained on large volumes of chest CT paired
with radiology reports, compresses each scan into 1152 numbers called an
embedding. Scans a radiologist would describe similarly land close together in
that space.

The workflow has two stages:

| Stage | Notebook | Hardware | Runtime | Produces |
|---|---|---|---|---|
| 1 | `Part1_Pillar0_Embeddings_to_CSV.ipynb` | GPU | ~10 s per scan | `embeddings_*.csv` |
| 2 | `Part2_Train_and_Compare_Task_Heads.ipynb` | CPU (laptop) | minutes | metrics, plots, comparisons |

The encoder is **frozen**: its weights are never modified. It acts as a fixed
translator from images to numbers. All learning happens in the small classifier
trained afterwards.

## Why it is built this way

**The two stages are separated because their costs differ by orders of
magnitude.** Encoding 956 scans takes about three hours on a GPU. Training a
classifier on the resulting numbers takes seconds on a laptop. Separating them
means the expensive step runs once and every subsequent experiment is cheap.

**The encoder never sees a label.** It produces the same 1152 numbers regardless
of what is being predicted. One embedding run therefore supports any number of
different findings: only the label column changes. This is the practical basis
for the claim that additional task-specific models cost very little once the
first is built.

**The interface between the stages is a CSV.** Three columns carry everything
that matters:

| Column | Meaning |
|---|---|
| `labels` | 0 or 1 |
| `embedding` | the 1152-number vector as a bracketed string |
| `file_name` | study identifier, used to align two encoders |

Part 1 also writes `subsets` (which predefined split a study belongs to, empty
in Path A) and `orig_depth` (slice count before padding). Part 2 uses both where
present. Because the contract is a plain CSV, a different encoder can be
substituted by regenerating the file and changing nothing downstream.

## The two paths

| | Path A | Path B |
|---|---|---|
| Dataset | CT-RATE | RSNA STR Pulmonary Embolism Detection |
| Contrast | Non-contrast chest CT | CTPA (contrast) |
| Format | NIfTI | DICOM |
| Labels | 18 report-derived findings | Study-level PE labels including `central_pe` |
| Size | streamed per scan | about 900 GB, pre-mounted |
| Run on | Google Colab | Kaggle Notebooks |
| Access | HuggingFace account, gated but free | Kaggle account, accept competition rules |
| Use it for | any of the 18 findings, quick pipeline checks | central PE, replicating a published cohort |
| Cannot do | pulmonary embolism, no PE label exists | anything other than PE |

**Path A limitation worth stating in any report.** CT-RATE is non-contrast chest
CT. Pulmonary embolism cannot be assessed on non-contrast studies, and CT-RATE
carries no PE label among its 18 abnormalities. Results from Path A validate the
pipeline and can support claims about the findings CT-RATE does label. They can
never be presented as PE results.

**Path B must run on Kaggle.** The RSNA archive is about 900 GB, which rules out
Google Drive (15 GB free) and any laptop. Kaggle pre-mounts the competition data
at `/kaggle/input`, so the download cost is zero. The AWS Open Data mirror is a
single zip behind controlled access and is not usable for per-study streaming.

## One-time setup

**Local environment for Part 2.** Apple Silicon needs a native environment, not
one running under Rosetta, or conda will hang:

```bash
conda config --set subdir osx-arm64
conda create -n med-imaging python=3.11
conda activate med-imaging
pip install numpy pandas scikit-learn scipy matplotlib jupyter
```

**HuggingFace access.** Pillar-0 is gated, so both paths need this.

1. Create an account at `huggingface.co/join`.
2. Visit `huggingface.co/YalaLab/Pillar0-ChestCT` and request access.
3. For Path A also visit `huggingface.co/datasets/ibrahimhamamci/CT-RATE` and
   accept the terms.
4. Create a **read** token at `huggingface.co/settings/tokens`.

**Kaggle, Path B only.**

1. Create an account and complete phone verification, which is required for API
   tokens and for notebook internet access.
2. Open `kaggle.com/competitions/rsna-str-pulmonary-embolism-detection`, go to
   the Rules tab, and accept. Nothing mounts until you do.
3. In the notebook: Add-ons > Secrets > Add secret, label `HF_TOKEN`, paste the
   HuggingFace token, and confirm it is attached to the notebook. Never paste a
   token into a cell.

## Path A: CT-RATE on Colab

1. Upload `Part1_Pillar0_Embeddings_to_CSV.ipynb` to Colab.
2. Runtime > Change runtime type > T4 GPU.
3. Section 1: set `PATH = "A"`, then set `A_LABEL` to any column in CT-RATE's
   multi-abnormality label file, and `A_N_SCANS` to the number of scans.
4. Run Section 2, then restart the session, then continue from Section 3.
5. Paste the HuggingFace token when prompted.
6. Run to the end.

Output lands on Google Drive so a disconnect loses nothing. Reconnect, re-run
Sections 1 to 7, and the loop resumes from the last completed scan.

**Choosing a scan count.** Precision on AUC is governed by the smaller class.
Using Hanley and McNeil's variance, a balanced set at an expected AUC near 0.80
needs roughly:

| 95% CI half-width | Total scans |
|---|---|
| ±0.05 | 302 |
| ±0.04 | 472 |
| ±0.03 | 836 |

Below about 200 total the interval is wide enough that the result only
demonstrates the pipeline works.

## Path B: RSNA on Kaggle

1. Create a new Kaggle notebook and paste in the cells from
   `Part1_Pillar0_Embeddings_to_CSV.ipynb`.
2. Right sidebar: Add Input > the RSNA competition, and Add Input > your encode
   list dataset. Accelerator > GPU T4 x2. Internet > On.
3. Section 1: set `PATH = "B"`.
4. Run the smoke test in Section 8 and check the reported time per scan.
5. **Save Version > Save and Run All (Commit) > Save**, then close the tab. A
   commit runs headless for up to 12 hours and persists `/kaggle/working`, so
   the laptop can be switched off. Progress appears under the Versions tab.
6. When it finishes, download the CSV from the Output tab and move it next to
   Part 2.

Roughly 10 seconds per study, so 956 studies takes about 2.7 hours. Free GPU
quota is 30 hours per week, so stop idle sessions from Your Work > Active
Events.

To resume a run that was killed at the 12-hour limit, attach the previous
version's output as an input dataset and point the resume read at it.

## Building an encode list for Path B

Path B reads a CSV naming which studies to encode. Columns:

| Column | Meaning |
|---|---|
| `file_name` | study identifier, also the RSNA `SeriesInstanceUID` |
| `labels` | 0 or 1 |
| `subsets` | comma-joined list of splits, for example `D3,D4` |
| `study` | RSNA `StudyInstanceUID`, the first path component |
| `series` | RSNA `SeriesInstanceUID`, the second path component |

RSNA stores scans at `train/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm`.

To build one from scratch, download only the labels, which is a small file:

```bash
kaggle competitions download -c rsna-str-pulmonary-embolism-detection -f train.csv
unzip train.csv.zip
```

```python
import pandas as pd
d = pd.read_csv("train.csv")
s = d.drop_duplicates("StudyInstanceUID")

pos = s[s.central_pe == 1]
neg = s[(s.central_pe == 0) & (s.leftsided_pe == 0) & (s.rightsided_pe == 0)]

# example: all positives plus a 1:3 negative sample
neg = neg.sample(len(pos) * 3, random_state=42)
cohort = pd.concat([pos, neg])

out = d[d.StudyInstanceUID.isin(cohort.StudyInstanceUID)] \
        .drop_duplicates("StudyInstanceUID")
out = out.rename(columns={"SeriesInstanceUID": "series",
                          "StudyInstanceUID": "study"})
out["file_name"] = out.series
out["labels"] = out.central_pe
out["subsets"] = ""      # or assign D1..D4 / TEST here
out[["file_name", "labels", "subsets", "study", "series"]] \
   .to_csv("p3_encode_list.csv", index=False)
```

Then upload it as a Kaggle Dataset so the notebook can mount it.

**Why 1:3.** The published cohort this pipeline reproduces used a 1:3
positive-to-negative ratio for its largest training set, chosen to match the
training size trends of other benchmarks for the Google CT Foundation model. Its
test sets are also 1:3, so training at the prevalence you evaluate at keeps the
design clean.

## Running Part 2

```bash
conda activate med-imaging
jupyter notebook Part2_Train_and_Compare_Task_Heads.ipynb
```

Section 1 is the only cell that normally needs editing.

**`MODE = "generic"`** makes one stratified random split. Use this for Path A
output or any CSV without predefined splits.

**`MODE = "replication"`** reads the `subsets` column and evaluates every
training subset against the same fixed test set. Use this for Path B output
built against a published cohort.

**`COMPARE_CSV`** optionally points at a second embedding file covering the same
studies. Part 2 aligns them on `file_name` and runs a paired DeLong test for
every head, which is how a backbone swap should be measured: same scans, same
splits, same head, one variable changed. Results go to
`comparison_backbones.csv` with Holm-corrected p values. Comparing a number
computed here against a number printed in a paper is not equivalent, because
preprocessing, head configuration and evaluation protocol all differ.

**`GRID_SEARCH`** controls how the heads are configured.

| | `False` (default) | `True` |
|---|---|---|
| Fits per head | 1 | up to 13,500 |
| Runtime | about a minute | hours |
| Use for | the backbone comparison | matching a published protocol |

Fixed hyperparameters are correct for a backbone comparison, because both
encoders receive the identical head and neither can be favoured by tuning. Grid
search is needed only when quoting a number alongside one from a paper that used
a search.

Note the direction of the bias. A published figure produced by grid search,
compared against a fixed-hyperparameter figure, understates the fixed one, since
tuning would likely improve it. That is the safe way to be wrong, but the
protocols are not identical until the grid is run.

`GRID_SPLITS` limits the search to named splits. The default `["D3"]` is the
largest balanced set and the natural primary comparison. When grid search is on,
each encoder is searched separately so neither is handicapped, and the selected
parameters are printed and saved.

### What a run produces

| File | Contents |
|---|---|
| `run_config.json` | every setting, plus row counts, embedding dimension, split sizes, signal check, and the Python and platform version |
| `metrics.csv` | AUC with interval, sensitivity and specificity per split and head |
| `head_comparisons.csv` | DeLong between heads within each split, Holm corrected |
| `comparison_backbones.csv` | DeLong between encoders, every head and split, Holm corrected |
| `best_params.json` | hyperparameters the grid selected, per head, split and encoder |
| `summary.txt` | the headline numbers as plain text, ready to paste into a message |
| `fig_auc_by_head.png`, `fig_scaling.png` | figures |

`run_config.json` is what makes a run reproducible later. It records the row
counts and embedding dimension actually loaded, not just the settings, so a
changed input file is detectable after the fact.

A typical day is two runs: one with `GRID_SEARCH = False` for the controlled
comparison, and one with `GRID_SEARCH = True, GRID_SPLITS = ["D3"]` to match a
published protocol. They land in separate folders and neither disturbs the
other.

## Reading the results

**The heads tie.** Expected, and the informative outcome. If a straight line
does as well as a neural network, the finding is already linearly separable in
the embedding, which means the frozen encoder is doing the diagnostic work and
the head is only reading off an answer that is present. That is the core
evidence for the frozen-encoder approach: one expensive pass over the imaging
data, then a trivial head.

**AUC rises with training size.** Expected. A flat curve near chance suggests the
finding is not represented in the embedding at all. A curve that plateaus early
suggests more labels will not help and the encoder is the limit.

**Specificity at the operating point matters more than AUC for deployment.** AUC
summarises every possible threshold; a deployed system needs one. A model with
good AUC and poor specificity at a clinically required sensitivity generates
enough false alarms to cause alert fatigue, which stops adoption regardless of
how the summary statistic looks.

**The signal check is not optional.** With more embedding dimensions than
training samples, a flexible model can separate almost any labelling by chance.
Section 8 shuffles the training labels and repeats the identical procedure. If
the real result does not sit far outside the shuffled range, nothing else in the
notebook means anything.

## Troubleshooting

**`Exporter config not found: video_hevc`**
RAVE resolves exporter configs relative to its package directory, which breaks
under a pip install. Section 4 of Part 1 rewrites the config with an absolute
path and switches to lossless `lz4_fast`.

**`No DICOM files found` and `GDCMSeriesFileNames: No Series can be found`**
RSNA de-identification replaced `SeriesInstanceUID` with a 12-character hash
rather than a dotted numeric UID. GDCM validates UIDs and discards the files;
pydicom does not. Part 1 reads the DICOMs with pydicom and writes a NIfTI, which
sidesteps the scanner entirely.

**`EinopsError: can't divide axis of length 22 in chunks of 4`**
The volume is shorter than 256 slices. Pillar-0's merge ratio requires the patch
grid to be divisible by 4. `to_tensor` pads short volumes with air, so this only
appears if that function is bypassed.

**`GatedRepoError: 401` on the model**
Access to `YalaLab/Pillar0-ChestCT` was not granted, or the token is missing.
Check the model page shows access granted, and that the Kaggle secret is
attached to this specific notebook.

**`403 Forbidden` from the Kaggle CLI**
Competition rules were not accepted.

**`OSError: Could not find kaggle.json`**
The CLI needs the downloaded `kaggle.json` at `~/.kaggle/kaggle.json` with mode
600. A bare token string in another file will not work. Also confirm `which
kaggle` points inside the intended environment.

**`AttributeError` on `extract_vision_feats`**
The model expects a modality dict, `{"chest_ct": tensor}`, not a bare tensor.

**Notebook cannot find `/kaggle/input/<competition-slug>`**
Newer Kaggle mounts nest one level deeper. Use
`glob.glob("/kaggle/input/competitions/*")` and
`glob.glob("/kaggle/input/datasets/**/<file>", recursive=True)`.

**Conda hangs on Apple Silicon**
The environment was created under Rosetta. Rebuild natively with
`conda config --set subdir osx-arm64`.

## Scope and limitations

**Single site.** Every result here comes from one data source. Embeddings encode
scanner and protocol characteristics alongside anatomy, so a drop at a second
institution is expected and must be measured before any claim generalises. The
published work this pipeline reproduces found that its headline problem on
external data was not the AUC drop but a collapse in specificity at the
transferred operating point, which threshold recalibration did not fix.

**Report-derived labels.** Both CT-RATE and the external cohorts commonly paired
with RSNA use labels extracted from radiology report text rather than expert
image-level adjudication. Label noise bounds achievable performance.

**Frozen encoder only.** Nothing here fine-tunes Pillar-0. Frozen embeddings plus
a task head is one point on a spectrum; fine-tuning updates the backbone, and
adapter methods sit in between. Pillar-0 is open, so all three are available,
which is the main practical difference from closed alternatives.

**Padding.** Volumes shorter than 256 slices at 1.25 mm are centre-padded with
air. On CTPA cohorts most studies are padded, since typical coverage is under
320 mm. `orig_depth` is recorded so this can be checked, and Part 2 tests whether
depth alone predicts the label.

## Reference

**Model.** Pillar-0, `YalaLab/Pillar0-ChestCT`. Agrawal et al. 2025,
arXiv:2511.17803.

**Preprocessing.** RAVE, `github.com/YalaLab/rave`, using
`configs/ct_chest.yaml`: resample to 1.25 mm isotropic, crop or pad to 256 x 256
in-plane, take the middle 256 slices, SimpleITK backend.

**Geometry.** Input is `(B, 11, 256, 256, 256)`. The 11 channels are the CT
windows returned by `rve.get_available_windows("CT")`: ten anatomical windows
plus minmax. This is fixed, not a free choice. The spatial dimensions come from
Pillar-0's own configuration and its learned position embeddings reject other
geometries.

**Pinned dependency.** `transformers==4.55.2`, installed last because RAVE pulls
in its own version. Newer versions fail to load this checkpoint through
`trust_remote_code`.

**Datasets.**
CT-RATE, `huggingface.co/datasets/ibrahimhamamci/CT-RATE`, gated but free.
RSNA STR Pulmonary Embolism Detection,
`kaggle.com/competitions/rsna-str-pulmonary-embolism-detection`.
Colak et al. 2021, Radiology: Artificial Intelligence 3(2):e200254.

**Statistics.**
Hanley and McNeil 1982, Radiology 143:29-36, for AUC variance and sample size.
DeLong et al. 1988, Biometrics 44:837-845, for comparing correlated ROC curves.
Sun and Xu 2014, IEEE SPL 21:1389-1393, for the fast implementation.
Holm 1979, Scand J Statist 6:65-70, for multiple-comparison correction.

## Repository layout

```
Part1_Pillar0_Embeddings_to_CSV.ipynb      CT scans to embeddings, run on Colab or Kaggle with a GPU
Part2_Train_and_Compare_Task_Heads.ipynb   Embeddings to trained classifiers and metrics, runs on a laptop
embeddings_pillar0_rsna.csv                Part 1 output for the RSNA cohort (Path B)
```

Running the notebooks also creates `embeddings_pillar0_ctrate.csv` (Path A),
`p3_encode_list.csv` (the Path B work list) and a `results/` folder with one
subfolder per Part 2 run, holding the files listed under What a run produces.

Part 2 expects the embedding CSV in the same folder as the notebook. Edit
`CSV_PATH` in Section 1 if it lives elsewhere.

Run folders are named `<timestamp>_<label>_<mode>_<fixed|grid>`, so the settings
are visible without opening anything. Nothing is ever overwritten.

## Getting the files

```bash
cd ~/Desktop
git clone https://github.com/peteryazdi/Pillar0_finetuning.git
```

This works in Terminal on macOS and Linux and in PowerShell on Windows. Part 2
runs locally after the one-time setup above; Part 1 needs a GPU on Colab or
Kaggle.

## Author

Peter Yazdi, [peteryazdi@gmail.com](mailto:peteryazdi@gmail.com)
