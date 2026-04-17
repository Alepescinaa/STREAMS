import argparse, os, sys
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.stats import truncnorm

torch.backends.cudnn.benchmark = True

# -----------------------------
# Dataset
# -----------------------------
class InferDataset(Dataset):
    def __init__(self, feather_path, covariate_cols, prior_col="onset_prior"):
        df = feather.read_feather(feather_path)

        required = ["patient_id", "a", "b", prior_col]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing required column '{c}' in {feather_path}")

        for c in covariate_cols:
            if c and c not in df.columns:
                raise ValueError(f"Missing covariate '{c}' in {feather_path}")

        self.df = df.reset_index(drop=True)

        # prior -> tensor (N,1)
        prior = df[prior_col].fillna(0.5).clip(1e-6, 1 - 1e-6).astype("float32").values
        self.x = torch.tensor(prior, dtype=torch.float32).unsqueeze(1)

        # cond -> tensor (N,C) or empty
        if len(covariate_cols):
            cond = df[covariate_cols].astype("float32").values
        else:
            cond = np.zeros((len(df), 0), dtype="float32")
        self.cond = torch.tensor(cond, dtype=torch.float32)

        # intervals
        self.a = torch.tensor(df["a"].values, dtype=torch.float32).unsqueeze(1)
        self.b = torch.tensor(df["b"].values, dtype=torch.float32).unsqueeze(1)

        # patient ids
        self.patient_id = df["patient_id"].astype(np.int64).values

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.cond[idx], self.a[idx], self.b[idx], int(self.patient_id[idx])


# -----------------------------
# Model
# -----------------------------
class Encoder(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim + cond_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ReLU()
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_lv = nn.Linear(64, latent_dim)

    def forward(self, x, cond):
        h = torch.cat([x, cond], dim=1)
        h = self.net(h)
        return self.fc_mu(h), self.fc_lv(h)


class Decoder(nn.Module):
    def __init__(self, latent_dim, cond_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),  nn.LayerNorm(64),  nn.ReLU()
        )
        self.fc_p  = nn.Linear(64, 1)
        self.fc_mu = nn.Linear(64, 1)
        self.fc_sd = nn.Linear(64, 1)

    def forward(self, z, cond, a, b):
        h = torch.cat([z, cond], dim=1)
        h = self.net(h)
        p_onset = torch.sigmoid(self.fc_p(h))
        age_mu  = a + F.softplus(self.fc_mu(h))
        age_sd  = F.softplus(self.fc_sd(h)) + 1e-6
        return p_onset, age_mu, age_sd


class CVAE(nn.Module):
    def __init__(self, x_dim, cond_dim, latent_dim):
        super().__init__()
        self.enc = Encoder(x_dim, cond_dim, latent_dim)
        self.dec = Decoder(latent_dim, cond_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, cond, a, b, deterministic: bool = False):
        mu, lv = self.enc(x, cond)
        if deterministic:
            z = mu
        else:
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            z = mu + eps * std
        p_onset, age_mu, age_sd = self.dec(z, cond, a, b)
        return p_onset, age_mu, age_sd, mu, lv


# -----------------------------
# Truncated Normal Sampler
# -----------------------------
def sample_truncated_normal(mu: np.ndarray, sd: np.ndarray,
                             a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Vectorised truncated normal draw.
    For each element: X ~ TruncNormal(mu, sd, a, b).
    If a >= b (degenerate interval) returns b.
    """
    sd = np.maximum(sd, 1e-6)
    b_safe = np.minimum(np.maximum(b - 1e-3, a + 1e-3), b)

    alpha = (a    - mu) / sd
    beta  = (b_safe - mu) / sd

    out = b.copy()
    valid = b_safe > a
    if valid.any():
        out[valid] = truncnorm.rvs(
            alpha[valid], beta[valid],
            loc=mu[valid], scale=sd[valid]
        )
    return out


# -----------------------------
# Inference: m stochastic draws
# -----------------------------
@torch.no_grad()
def predict(model, loader, device, m: int = 20, seed: int = 0):
    """
    For each patient draw m samples from the latent posterior z ~ N(mu, sigma).
    Each draw j gives (p_onset_ij, age_mu_ij, age_sd_ij) from the decoder.
    Then:
      disease_status_ij ~ Bernoulli(p_onset_ij)
      if disease_status_ij == 1:
          disease_age_ij ~ TruncNormal(age_mu_ij, age_sd_ij, a_i, b_safe_i)
      else:
          disease_age_ij = b_i

    Returns a long-format DataFrame with columns:
      patient_id, draw, p_onset, age_mu, age_sd, disease_status, disease_age
    """
    rng = np.random.default_rng(seed)
    model.eval()

    # Accumulate per-draw tensors: list over j=1..m, each entry (N_batch,)
    # We'll store lists of arrays indexed [draw][batch]
    all_draws = {
        "patient_id": [],   # collected once from first draw
        "a": [],
        "b": [],
        "p_onset":  [[] for _ in range(m)],
        "age_mu":   [[] for _ in range(m)],
        "age_sd":   [[] for _ in range(m)],
    }
    ids_collected = False

    for x, cond, a, b, pid in loader:
        x, cond, a, b = x.to(device), cond.to(device), a.to(device), b.to(device)

        if not ids_collected:
            pass  # collect below

        pid_arr = np.array(pid, dtype=np.int64) if isinstance(pid, list) else np.asarray(pid, dtype=np.int64)
        a_arr   = a.squeeze(1).cpu().numpy()
        b_arr   = b.squeeze(1).cpu().numpy()

        all_draws["patient_id"].append(pid_arr)
        all_draws["a"].append(a_arr)
        all_draws["b"].append(b_arr)

        for j in range(m):
            p_j, mu_j, sd_j, _, _ = model(x, cond, a, b, deterministic=False)
            all_draws["p_onset"][j].append(p_j.squeeze(1).cpu().numpy())
            all_draws["age_mu"][j].append(mu_j.squeeze(1).cpu().numpy())
            all_draws["age_sd"][j].append(sd_j.squeeze(1).cpu().numpy())

    # Concatenate batches
    pids  = np.concatenate(all_draws["patient_id"])   # (N,)
    a_all = np.concatenate(all_draws["a"])             # (N,)
    b_all = np.concatenate(all_draws["b"])             # (N,)
    N = len(pids)

    p_mat  = np.stack([np.concatenate(all_draws["p_onset"][j]) for j in range(m)], axis=1)   # (N, m)
    mu_mat = np.stack([np.concatenate(all_draws["age_mu"][j])  for j in range(m)], axis=1)   # (N, m)
    sd_mat = np.stack([np.concatenate(all_draws["age_sd"][j])  for j in range(m)], axis=1)   # (N, m)

    # Sample disease_status and disease_age
    u = rng.uniform(size=(N, m))
    status_mat = (p_mat > u).astype(np.int32)          # (N, m)  1=onset 0=no onset

    age_mat = np.full((N, m), np.nan, dtype=np.float32)

    for j in range(m):
        onset_idx  = np.where(status_mat[:, j] == 1)[0]
        no_onset_idx = np.where(status_mat[:, j] == 0)[0]

        # no-onset: age = b
        age_mat[no_onset_idx, j] = b_all[no_onset_idx]

        # onset: sample from truncated normal
        if len(onset_idx):
            age_mat[onset_idx, j] = sample_truncated_normal(
                mu=mu_mat[onset_idx, j],
                sd=sd_mat[onset_idx, j],
                a=a_all[onset_idx],
                b=b_all[onset_idx],
            )

    # Build long-format DataFrame  (N * m rows)
    rep_pids   = np.repeat(pids,  m)                              # (N*m,)
    draws      = np.tile(np.arange(1, m + 1), N)                  # (N*m,)
    rows = pd.DataFrame({
        "patient_id":     rep_pids,
        "draw":           draws,
        "p_onset":        p_mat.ravel(order="C").astype(np.float32),
        "age_mu":         mu_mat.ravel(order="C").astype(np.float32),
        "age_sd":         sd_mat.ravel(order="C").astype(np.float32),
        "disease_status": status_mat.ravel(order="C").astype(np.int32),
        "disease_age":    age_mat.ravel(order="C").astype(np.float32),
    })

    return rows


def save_table(path, df):
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        feather.write_feather(df, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint",      type=str, help="Path to best checkpoint (.pt)")
    ap.add_argument("infer_feather",   type=str, help="Feather with patient_id, a, b, onset_prior and covariates")
    ap.add_argument("covariate_str",   type=str, help="Comma-separated covariate names (same as training)")
    ap.add_argument("--latent_dim",    type=int, default=5,    help="Must match training")
    ap.add_argument("--out",           type=str, default="predictions.feather")
    ap.add_argument("--use_student",   action="store_true",    help="Use student weights instead of EMA teacher")
    ap.add_argument("--batch_size",    type=int, default=256)
    ap.add_argument("--m",             type=int, default=20,   help="Number of stochastic latent draws per patient")
    ap.add_argument("--seed",          type=int, default=0,    help="RNG seed for sampling")
    ap.add_argument("--cpu",           action="store_true")
    args = ap.parse_args()

    covs = [c.strip() for c in args.covariate_str.split(",") if c.strip()]
    ds = InferDataset(args.infer_feather, covs)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    x_dim, cond_dim = 1, len(covs)
    model = CVAE(x_dim, cond_dim, args.latent_dim).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and ("teacher" in ckpt or "student" in ckpt):
        state = ckpt["student"] if args.use_student else ckpt["teacher"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=True)

    print(f"Running inference: m={args.m} stochastic draws per patient ...")
    rows = predict(model, dl, device, m=args.m, seed=args.seed)

    save_table(args.out, rows)
    print(f"Wrote predictions to {args.out}  (rows: {len(rows)}, patients: {rows['patient_id'].nunique()})")


if __name__ == "__main__":
    main()
