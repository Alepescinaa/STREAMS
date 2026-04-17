# STREAMS
Semi-supervised Trajectory REconstruction Across interval-censored Multi-State models.

`STREAMS` is an R package for reconstructing latent disease trajectories in
progressive illness–death models using semi-supervised learning and then fitting
parametric multi-state models with proper pooling across multiple imputations.

## Installation

install.packages("devtools")  # if not already installed
devtools::install_github("Alepescinaa/STREAMS")


library(STREAMS)

## Load example panel data
data("toy_example")
panel_data <- toy_example

## Covariates to be used in the model
cov_vector <- c("cov1", "cov2", "cov3")

## Sanity check on the input structure (optional, it is nested in main pipeline)
check_input_data(panel_data)

## Run the main pipeline

fit_streams <- run_streams(
  data              = panel_data,
  cov_vector        = cov_vector,
  python            = Sys.which("python3"),
  pu_args           = list(verbose = TRUE),
  features_prop_add = NULL
)

##Overall object
class(fit_streams)
summary(fit_streams, plots = TRUE)

## First transition model
trans1_streams <- fit_streams[[1L]]
class(trans1_streams)

print(trans1_streams)
confint(trans1_streams)
attr(trans1_streams, "rubin") 
coef(trans1_streams)
summary(trans1_streams)


