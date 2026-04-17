#' @noRd
.default_pu_args <- list(
  max_iter = 1000,
  tol = 1e-3,
  clip = 1e-3,
  damp = 0.3,
  shrink_k = 0.0,
  verbose = FALSE
)
#' @noRd
.default_cvae_args <- list(
  latent_dim = 5,
  max_epochs = 200,
  batch_size = 256,
  lr = 2e-3,
  r_all = 0.3,
  lambda_fix_max = 1.0,
  fix_ramp_epochs = 50,
  K_aug = 5,
  prior_aug = "beta",
  early_stop_patience = 15,
  early_stop_warmup = 50
)
#' @noRd
.default_infer_args <- list(
  latent_dim = 5,
  mc_samples = 0,
  batch_size = 256
)

#' @noRd
.default_infer_args2 <- list(
  latent_dim = 5,
  m = 1,
  batch_size = 256
)
