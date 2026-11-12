# Data

All data are public and none are stored in this repository.

| Dataset | Used for | Access |
|---|---|---|
| HCP Young Adult 3T diffusion and surfaces | Reference, simulated protocols, training | Free account and data use terms, then S3 keys |
| HCP 7T retinotopy (pRF results, OSF bw9ec) | Measured retinotopy of the test subjects | Open |
| HCP 7T diffusion, HCP Retest | External test, reproducibility | Same S3 keys |
| Natural Scenes Dataset (NSD) | External test on other people and another scanner | Open S3 bucket after the NSD access form |
| Benson 2014 atlas (neuropythy), HCP standard spheres | Atlas key; carrying maps between meshes | Open, from GitHub |

## Access

1. Create an account at https://balsa.wustl.edu and accept the WU-Minn HCP Open Access Data Use Terms.
2. On the HCP Young Adult page, request AWS S3 access to get an access key and a secret key (free).
3. `cp .env.example .env` and fill in `S3ACCESS_KEY` and `S3SECRET_KEY`. `.env` is ignored by git.
4. For NSD, submit the access form at https://naturalscenesdataset.org. No key is needed to download.

## Download

```bash
bash scripts/download_data.sh        # full configuration
bash scripts/download_data.sh dev    # dev subjects only
```

Builds the subject lists, prints how many files and gigabytes are still missing, then downloads in parallel with a
progress bar (`WORKERS=16` for more connections). Complete files are skipped and time-outs are retried, so it can
be stopped and started again. Sizes: about 1.4 GB per HCP 3T subject, 1.1 GB per 7T scan,
4.1 GB per NSD run, 1 GB of retinotopy results; about 336 GB (2,231 files) for the default full configuration.

## Files read

- **HCP** (`HCP_1200/<id>/T1w/`, `HCP_Retest/`): `Diffusion/` or `Diffusion_7T/` (`data.nii.gz`, `bvals`, `bvecs`,
  `nodif_brain_mask.nii.gz`), `aparc+aseg.nii.gz`, white and pial `*_MSMAll.32k_fs_LR.surf.gii`; for the test
  subjects also `very_inflated_MSMAll.32k_fs_LR.surf.gii` (about 0.7 MB per hemisphere, Figure 1 only).
- **Retinotopy**: `prfresults.mat` (pRF maps of 181 subjects and group averages, all runs and each half) and
  `atlas.mat` (Wang 2015 visual areas).
- **NSD**: diffusion runs 1 and 2 in the 0.8 mm space, native surfaces, `aseg_0pt8`, pRF maps, `prf-visualrois`,
  `sphere.reg`.

```
data/hcp/   data/nsd/   raw files, same layout as the buckets
data/osf/   retinotopy results, atlas and sphere files
data/cache/ one folder per scan: crop, fODFs, tractograms, scores
```

## Terms

- **HCP:** do not redistribute. Acknowledge: "Data were provided by the Human Connectome Project, WU-Minn
  Consortium (Principal Investigators: David Van Essen and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH
  Institutes and Centers that support the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for
  Systems Neuroscience at Washington University."
- **NSD:** research use only, no redistribution. Cite Allen et al. (2022) and acknowledge NSF IIS-1822683 and NSF
  IIS-1822929.
- Never commit raw data or per-subject derived files.
