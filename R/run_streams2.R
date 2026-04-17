#' Run STREAMS: Main function
#'
#' @description
#' \code{run_streams()} runs the full STREAMS pipeline starting from longitudinal visit-level data:
#' \enumerate{
#'   \item Builds a patient-level dataset and PU-learning priors via \code{\link{prepare_data}}.
#'   \item Trains a Conditional VAE (with Mean-Teacher style training) via the packaged Python script
#'         \code{train.py} (called through \code{\link{streams_python}}).
#'   \item Runs model inference via \code{inference.py} to obtain \code{m} stochastic draws per
#'         patient from the latent posterior. Each draw yields onset probability, onset-age
#'         distribution parameters, a sampled disease status, and a sampled disease age
#'         (from a truncated normal if onset = 1, else b).
#'   \item Fits \code{m} parametric illness-death multi-state models via \code{\link{fit_model}},
#'         one per draw.
#'   \item Aggregates parameters across imputations via \code{\link{averaging_params}}.
#' }
#'
#' @details
#' The pipeline exchanges data between R and Python using Feather files
#' and stores a trained model checkpoint. See the "Side effects" section below.
#'
#' @param data A `data.table` or `data.frame` that must contain the following columns:
#'   - `patient_id`: Unique identifier for each patient (numeric).
#'   - `dead`: Binary indicator (0/1) for whether the patient is dead.
#'   - `death_time`: Time of death if it has occurred or censoring time otherwise (numeric).
#'   - `onset`: Binary indicator (0/1) for disease onset.
#'   - `onset_age`: Age at disease onset if it has occurred or `death_time` otherwise (numeric).
#'   - `age`: Patient's current age at that specific visit (numeric).
#'   - `visits`: Indicator of the current visit (numeric).
#'   The `data.frame` can contain extra columns with covariate values.
#'
#' @param cov_vector Either:
#'   \itemize{
#'     \item A character vector of covariate names to be used for all transitions; or
#'     \item A named list of character vectors, one per transition, with names
#'       \code{"0->1"}, \code{"0->2"}, and \code{"1->2"}, specifying
#'       covariates for each transition separately.
#'   }
#'   In all cases, the union of covariates is passed to \code{prepare_data()},
#'   and used as conditioning vector in the variational autoencoder
#'   while \code{fit_model()} receives the per-transition specification.
#'   These are automatically scaled/encoded in the process.
#' @param m Integer. Number of stochastic latent draws (and therefore fitted multi-state models)
#'   contributing to the pooled estimates. Passed to both \code{inference.py} and \code{fit_model}.
#' @param clock_assumption Character. Time-scale assumption for the multi-state model. Passed to
#'   \code{\link{fit_model}}. Accepted values are \code{"forward"} to fit a Markov process or \code{"mix"} for a Semi-Markov process.
#' @param distribution A character string specifying the parametric form of baseline hazards.
#'   Must be one of the distributions available in `flexsurv::flexsurvreg`, e.g., `"weibull"`, `"exponential"`, `"gompertz"`.
#' @param custom_formula An optional specification of survival formulas:
#'   \itemize{
#'     \item If \code{NULL} (default), formulas are constructed automatically from
#'           \code{cov_vector}, allowing different covariates per transition (assuming linear effects).
#'     \item If a single \code{Surv()} formula, it is used for all transitions in the multi-state fit.
#'     \item If a list of three \code{Surv()} formulas, each element is used for
#'           the corresponding transition (1, 2, 3).
#'           }
#' @param lab_prop Numeric in (0, 1). Controls the PU-learning thresholding used to derive soft labels for training:
#'   among patients with \code{onset == 0}, those below the \code{lab_prop}-quantile of PU risk scores are treated
#'   as reliable negatives (\code{onset_soft = 0}); remaining \code{onset == 0} are left unlabeled.
#' @param features_prop_add Optional character vector of additional selection (propensity) features to append
#'   to the fixed default selection features used inside \code{\link{pu_learning}}.
#'   Use this only for covariates that plausibly affect the observation/labeling process.
#'   \strong{Warning:} adding variables overlapping with \code{cov_vector}
#'   can induce leakage/identifiability issues in SAR-PU and may destabilize EM updates.
#'
#' @param pu_args Named list of PU-learning hyperparameters overriding \code{.default_pu_args} and forwarded to
#'   the PU-learning routine. Supported keys:
#'   \describe{
#'     \item{max_iter}{Maximum number of EM iterations for PU learning.}
#'     \item{tol}{Convergence tolerance on successive log-likelihood differences.}
#'     \item{clip}{Probability clipping threshold for numerical stability.}
#'     \item{damp}{Damping factor in (0,1) for updating PU components to reduce oscillations.}
#'     \item{shrink_k}{Non-negative shrinkage parameter to penalize regions with low labeling propensity.}
#'     \item{verbose}{Logical. If \code{TRUE}, prints PU-learning training diagnostics.}
#'   }
#'
#' @param cvae_args Named list of training hyperparameters overriding \code{.default_cvae_args}.
#'   These entries are converted to command-line flags and passed to the Python training script \code{train.py}.
#'   Common keys in STREAMS are:
#'   \describe{
#'     \item{latent_dim}{Latent dimension of the CVAE.}
#'     \item{max_epochs}{Maximum number of training epochs.}
#'     \item{batch_size}{Mini-batch size.}
#'     \item{lr}{Learning rate.}
#'     \item{K_aug}{Number of independent prior augmentations evaluated by the \emph{teacher} per unlabeled
#'     input to estimate a mean teacher score \eqn{\bar p_i} and its variance \eqn{v_i}. Increasing \code{K_aug}
#'     yields a more stable uncertainty estimate but increases compute.}
#'     \item{r_all}{Numeric in \eqn{[0,1]}. Global \emph{selection rate} in the proportion-based pseudo-labeling mode
#'     (default). Among unlabeled items that pass the uncertainty gate, the trainer targets pseudo-labeling
#'     approximately an \code{r_all} fraction per batch by setting adaptive thresholds on teacher mean scores \eqn{\bar p_i}.}
#'     \item{lambda_fix_max}{ Maximum weight of the FixMatch-style consistency term. The effective weight is ramped up with epoch
#'     and capped at \code{lambda_fix_max}, so the student first stabilizes on labeled losses before trusting
#'     teacher pseudo-labels.}
#'     \item{fix_ramp_epochs}{Number of epochs used for the monotone ramp-up schedule.}
#'     \item{prior_aug}{Character. Type of prior augmentation used to create stochastic, distribution-matched views
#'     for student and teacher (\code{"beta"}, \code{"gauss"} or \code{"dropout"}). The same augmentation family is used for both networks,
#'     but sampled independently to generate noisy yet matched inputs for consistency.}
#'     \item{early_stop_patience}{Early-stopping patience (number of epochs without improvement).}
#'     \item{early_stop_warmup}{Number of warm-up epochs before early stopping is enabled.}
#'   }
#'   Any additional entries are forwarded as \code{--key value} flags; logical \code{TRUE} entries are forwarded
#'   as \code{--key}.
#'
#' @param infer_args Named list of inference hyperparameters overriding \code{.default_infer_args}.
#'   These entries are converted to command-line flags and passed to the Python inference script \code{inference.py}.
#'   Common keys in STREAMS are:
#'   \describe{
#'     \item{latent_dim}{Latent dimension used at inference (should match the trained model).}
#'     \item{batch_size}{Mini-batch size used at inference.}
#'     \item{seed}{RNG seed for stochastic draws and truncated-normal sampling in Python.}
#'   }
#'   Note: \code{m} is forwarded automatically from the top-level argument; do not set it here.
#'
#' @param python Character. Path to the Python executable used to run \code{train.py} and \code{inference.py}.
#'   Defaults to \code{Sys.which("python")}; if empty/NULL, a fallback such as \code{"python3"} may be used.
#' @param out_dir Character. Directory used to store intermediate artifacts (Feather inputs/outputs, logs, model checkpoint).
#'   If \code{NULL}, a unique timestamped subdirectory under \code{tempdir()} is created.
#' @param keep_intermediate Logical. If \code{TRUE}, \code{out_dir} is kept after the run; if \code{FALSE}, \code{out_dir}
#'   is deleted at the end.
#' @param n_cores Integer. Number of cores used to fit the \code{m} multi-state models in parallel
#'   via \code{parallel::mclapply}. (Note: \code{mclapply} is not supported on Windows.)
#' @param seed Integer. For full end-to-end reproducibility. Passed to both Python (RNG for sampling) and R.
#'
#' @return
#' A Rubin-pooled parametric multi-state model fit based on \code{m} imputations.
#'
#' Pooled estimates and uncertainty are available through \code{coef()}, \code{vcov()},
#' \code{confint()}, and \code{summary(x, coefs = TRUE)}.
#'
#' @seealso \code{\link{prepare_data}}, \code{\link{pu_learning}}, \code{\link{streams_python}},
#'   \code{\link{fit_model}}
#'
#' @examples
#' \dontrun{
#' est <- run_streams(
#'   data = df_panel,
#'   cov_vector = c("sex", "bmi", "smoking"),
#'   m = 20,
#'   clock_assumption = "forward",
#'   distribution = "gompertz",
#'   lab_prop = 0.15,
#'   pu_args = list(max_iter = 1000, tol = 1e-5),
#'   cvae_args = list(max_epochs = 200, latent_dim = 5, prior_aug = "beta"),
#'   out_dir = NULL,
#'   keep_intermediate = TRUE,
#'   n_cores = 4,
#'   seed = 42
#' )
#' }
#'
#' @importFrom utils modifyList
#' @export

run_streams2 <- function(
    data,
    cov_vector,

    # --- Core analysis ---
    m = 20,
    clock_assumption = "forward",
    distribution = "gompertz",
    custom_formula = NULL,

    # --- PU learning ---
    lab_prop = 0.5,
    pu_args = list(),
    features_prop_add = NULL,

    # --- CVAE / FixMatch (macro knobs) ---
    cvae_args = list(),

    # --- Inference ---
    infer_args = list(),

    # --- Execution ---
    python = Sys.which("python"),
    out_dir = NULL,
    keep_intermediate = FALSE,
    n_cores = 4,
    seed = 42
)
{

  # --- merge default arguments
  pu_cfg    <- modifyList(.default_pu_args,   pu_args)
  cvae_cfg  <- modifyList(.default_cvae_args, cvae_args)
  infer_cfg <- modifyList(.default_infer_args2, infer_args)

  cvae_cfg$seed  <- seed

  # m is always driven by the top-level argument; do not let infer_args override it
  infer_cfg$m    <- m
  infer_cfg$seed <- seed

  # normalize covariate specification
  normalize_cov_spec <- function(cov_spec) {
    trans_names <- c("0->1", "0->2", "1->2")

    if (is.character(cov_spec)) {
      cov_list <- lapply(trans_names, function(x) cov_spec)
      names(cov_list) <- trans_names
      return(cov_list)
    }

    if (is.list(cov_spec)) {
      if (is.null(names(cov_spec))) {
        stop("If cov_vector is a list, it must be named with transitions '0->1', '0->2', '1->2'.")
      }
      cov_list <- vector("list", length(trans_names))
      names(cov_list) <- trans_names

      for (tr in trans_names) {
        if (!is.null(cov_spec[[tr]])) {
          cov_list[[tr]] <- as.character(cov_spec[[tr]])
        } else {
          cov_list[[tr]] <- character(0)
        }
      }
      return(cov_list)
    }

    stop("cov_vector must be a character vector or a named list.")
  }

  cov_list  <- normalize_cov_spec(cov_vector)
  cov_union <- sort(unique(unlist(cov_list)))


  # --- helper: list -> CLI args vector
  as_cli_args <- function(arg_list) {
    if (length(arg_list) == 0) return(character(0))
    out <- character(0)

    for (k in names(arg_list)) {
      v <- arg_list[[k]]

      if (is.logical(v)) {
        if (isTRUE(v)) out <- c(out, paste0("--", k))
        next
      }

      out <- c(out, paste0("--", k), as.character(v))
    }

    out
  }


  # --- folders for temp results
  if (is.null(out_dir)) {
    out_dir <- file.path(tempdir(), sprintf("STREAMS_%s", format(Sys.time(), "%Y%m%d_%H%M%S")))
  }

  if (!keep_intermediate) {
    on.exit(unlink(out_dir, recursive = TRUE, force = TRUE), add = TRUE)
  }

  dirs <- list(
    data    = file.path(out_dir, "data"),
    models  = file.path(out_dir, "models"),
    results = file.path(out_dir, "results")
  )

  invisible(lapply(dirs, dir.create, recursive = TRUE, showWarnings = FALSE))

  input_path_train  <- file.path(dirs$data,    "training_data.feather")
  input_path_infer  <- file.path(dirs$data,    "inference_data.feather")
  train_split       <- file.path(dirs$data,    "train_split.feather")
  val_split         <- file.path(dirs$data,    "val_split.feather")
  model_path        <- file.path(dirs$models,  "best_model.pt")
  distributions_path <- file.path(dirs$results, "distributions.feather")
  logs_path         <- file.path(dirs$results,  "logs.feather")


  # --- check and prepare input data
  data <- check_input_data(data, cov_union)

  temp <- prepare_data(
    data              = data,
    cov_vector        = cov_union,
    lab_prop          = lab_prop,
    pu_args           = pu_cfg,
    train_path        = input_path_train,
    infer_path        = input_path_infer,
    features_prop_add = features_prop_add
  )
  cleaned_data <- temp[[1]]
  cov_str      <- paste(temp[[2]], collapse = ",")

  if (is.null(python)) python <- "python3"

  # --- TRAINING PY
  train_cli <- c(
    model_path,
    input_path_train,
    train_split,
    val_split,
    logs_path,
    cov_str,
    as_cli_args(cvae_cfg)
  )
  streams_python("train.py", args = train_cli, python = python)


  # --- INFERENCE PY
  # Python now handles all m stochastic draws, Bernoulli sampling of disease_status,
  # and truncated-normal sampling of disease_age. Output is a long-format feather:
  #   patient_id | draw | p_onset | age_mu | age_sd | disease_status | disease_age
  infer_cli <- c(
    model_path,
    input_path_infer,
    cov_str,
    "--out", distributions_path,
    as_cli_args(infer_cfg)
  )
  streams_python("inference2.py", args = infer_cli, python = python)


  # --- read long-format predictions (N_patients * m rows)
  distributions <- arrow::read_feather(distributions_path)

  # Sanity check: we expect exactly m draws per patient
  draws_per_patient <- unique(table(distributions$patient_id))
  if (!all(draws_per_patient == m)) {
    warning(sprintf(
      "Expected %d draws per patient but found: %s",
      m, paste(draws_per_patient, collapse = ", ")
    ))
  }


  # --- saving plot logs
  if (file.exists(logs_path)) {
    logs      <- arrow::read_feather(logs_path)
    logs_cols <- names(logs)

    plots_raw      <- plot_streams_total_and_fixmatch(logs, fixmatch_line = "raw",      print_plots = FALSE)
    plots_weighted <- plot_streams_total_and_fixmatch(logs, fixmatch_line = "weighted", print_plots = FALSE)

    loss_plots <- list(raw = plots_raw, weighted = plots_weighted)
  }


  # --- fit m multi-state models, one per draw
  # disease_status and disease_age for each draw j are read directly from
  # the Python output;
  if (.Platform$OS.type == "windows" && n_cores > 1) {
    warning("mclapply is not supported on Windows; using n_cores = 1.")
    n_cores <- 1
  }

  all_fits <- parallel::mclapply(1:m, function(j) {

    # Extract draw j for all patients, preserving cleaned_data row order
    draw_j <- distributions[distributions$draw == j, ]

    # Match to cleaned_data by patient_id to guarantee alignment
    draw_j <- draw_j[match(cleaned_data$patient_id, draw_j$patient_id), ]

    temp <- cleaned_data

    # Patients where this draw sampled onset = 1
    idx <- which(draw_j$disease_status == 1)

    if (length(idx)) {
      temp$onset[idx]     <- 1L
      temp$onset_age[idx] <- draw_j$disease_age[idx]
    }

    fit_model(temp, cov_list, clock_assumption, distribution, custom_formula)

  }, mc.cores = n_cores)


  # --- Pooling with Rubin's rules
  pooled_fit <- pool_rubin_all_transitions(
    all_fits, cl = 0.95, distribution, clock_assumption,
    cov_vector, custom_formula, loss_plots, logs_cols
  )

  return(pooled_fit)
}
