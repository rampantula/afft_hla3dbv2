# AFFT-HLA3DBv2

AFFT-HLA3DBv2 is an AlphaFold2 fine-tuning workflow for **peptide–HLA class I (pHLA-I) structure prediction**. The model was fine-tuned using 9-mer peptide/HLA-I structures from HLA3DB, with a dataset cutoff date of **July 1, 2026**.

Fine-tuning was conducted by **Tianjian Liang** and **Ram Pantula**. For questions or issues, please contact **Dr. Nikolaos Sgourakis**.

This branch of the repository contains the scripts needed to finetune the model.

## Overview

AFFT-HLA3DBv2 introduces several innovations designed for peptide–HLA-I structure modeling:

### 1. Combined MHC and peptide sequence similarity for template selection

BLOSUM62 similarity scores are calculated separately for the MHC and peptide sequences between each target and candidate template.

MHC and peptide scores are normalized independently using min–max normalization:

```text
normalized_score = (x - min(x)) / (max(x) - min(x))
```

The normalized scores are then combined as:

```text
total_score = weight_peptide × peptide_score + weight_MHC × MHC_score
```

The **top four templates** are selected for AlphaFold2 input.

### 2. Prevention of target leakage and redundant template selection

To reduce leakage and redundancy during template selection:

- The ground-truth structure of the input target is excluded from template selection.
- Redundant chain pairs originating from the same PDB entry are also excluded.

### 3. D-score–augmented fine-tuning loss

A peptide-backbone **D-score** term is incorporated into the AlphaFold2 fine-tuning objective:

```text
total_loss = AlphaFold_structure_loss + weight_dscore × D-score_loss
```

For the definition and interpretation of the D-score, see the HLA3DB publication:

https://www.nature.com/articles/s41467-023-42163-z

---

## Installation

### 1. Create the Python environment

Create the Conda environment from the provided environment file:

```bash
conda env create --file alphafold.yml
```

Using **Mamba** instead of Conda is recommended for faster dependency resolution.

### 2. Install AlphaFold2 model parameters

Follow the official AlphaFold installation instructions:

https://github.com/google-deepmind/alphafold

Download the AlphaFold parameter archive `alphafold_params_2022-12-06.tar`, extract it, and place the parameter files inside a `params/` directory.

---

## Data Preparation

### 1. Prepare the input Excel file

Input sequences are provided in an Excel file following the format of `Training_setv2.xlsx`.

The workflow uses the following information to construct each target and search for templates:

- Target PDB ID
- MHC sequence
- Peptide sequence

### 2. Generate template alignments

The `Template/` directory contains PDB structures that can be selected as templates. Generate the template-alignment files with:

```bash
python gen_align_MHC_Pep_unique_pdb.py \
    --excel Training_setv2.xlsx \
    --template-pdb-dir ./Template \
    --out-alignments-dir ./New_Align3 \
    --top-n 4
```

This step ranks candidate templates using the combined MHC/peptide sequence-similarity score and selects the **top four templates** for each target.

### 3. Generate the training/testing TSV

Convert the Excel input into the TSV format used by the AlphaFold fine-tuning and prediction scripts:

```bash
python generate_training_tsv.py \
    --excel Training_setv2.xlsx \
    --out training_datasetv23.tsv \
    --alignments-dir ./New_Align3
```

The generated TSV contains the target sequence and the corresponding template-alignment file for each target.

---

## Fine-Tuning

Fine-tune AlphaFold2 on the peptide–HLA-I training set with the D-score–augmented loss:

```bash
python dscore_loss_updatedv3.py \
    --data_dir ./ \
    --train_dataset ./training_datasetv23.tsv \
    --valid_dataset ./testing_datasetv23.tsv \
    --dump_valid_pdbs True \
    --valid_pdb_dir ./output \
    --anchor_class_file ./anchor_class.csv \
    --outprefix testrun \
    --dscore_weight 0.1 \
    --lr_coef 0.025 \
    --save_steps 391
```

---

## Structure Prediction

Use a fine-tuned parameter checkpoint to predict peptide–HLA-I structures:

```bash
python run_prediction.py \
    --targets testing_datasetv23.tsv \
    --params_file ./model/14298113_params_1564.pkl \
    --outfile_prefix test \
    --output_dir ./output
```

---

## Validation / Model Evaluation

To reproduce the validation-style evaluation workflow, run:

```bash
python run_predictionv3.py \
    --targets testing_dataset.tsv \
    --params_file ./model/14298113_params_1564.pkl \
    --outfile_prefix testrun_diagnostic \
    --output_dir ./output \
    --model_name model_2_ptm \
    --anchor_class_file ./anchor_class.csv \
    --exact_validation
```

> **Note:** `--exact_validation` is intended for validation/evaluation when the corresponding native structures and native alignments are available. For prediction of new structures without native experimental structures, use the **Structure Prediction** workflow above instead.

---

## Benchmark Results

### Benchmark setup

To minimize data leakage, pHLA-I structures deposited **after 2022** were held out from fine-tuning and used as an independent test set.

AFFT-HLA3DBv2 was benchmarked against the following baseline methods on the **same test set**:

- AlphaFold3
- Boltz-2
- ESMFold2
- AlphaFold2

### Evaluation metric

Prediction accuracy was evaluated using the peptide-backbone **D-score**, following the definition introduced in the HLA3DB study.

A prediction was considered **structurally accurate** when:

```text
D-score < 1.5
```

relative to the experimentally determined structure.

### Overall performance

On this benchmark, **AFFT-HLA3DBv2 achieved the highest overall success rate** among the evaluated methods.

Here, **success rate** refers to the percentage of test targets whose predicted peptide backbone has a D-score below 1.5 relative to the experimentally determined structure.

### Subgroup analysis

Benchmark results were additionally stratified by HLA type and peptide-backbone conformation:

| Subgroup | Definition |
| --- | --- |
| **A02** | Targets belonging to the A02 supertype |
| **Δ7-1** | Targets assigned to the Δ7-1 discrete peptide-backbone conformation |

In the HLA3DB structural classification, **Δ7** denotes an anchor class, while **Δ7-1** denotes a recurrent discrete peptide-backbone conformation within that class.

---

## Citation

If you use AFFT-HLA3DBv2 in your research, please cite:

> Gupta S, Nerli S, Kutti Kandy S, Mersky GL, Sgourakis NG. **HLA3DB: comprehensive annotation of peptide/HLA complexes enables blind structure prediction of T cell epitopes.** *Nature Communications*. 2023;14:6349. doi:10.1038/s41467-023-42163-z.
