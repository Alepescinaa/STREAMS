
library(STREAMS)


data("toy_example")
panel_data <- toy_example

cov_vector <- c("cov1", "cov2", "cov3")


check_input_data(panel_data)


fit_streams <- run_streams(
  data              = panel_data,
  cov_vector        = cov_vector,
  python            = Sys.which("python"),
  pu_args           = list(verbose = TRUE),
  features_prop_add = NULL
)

class(fit_streams)

trans1_streams <- fit_streams[[1L]]
trans2_streams <- fit_streams[[2L]]
trans3_streams <- fit_streams[[3L]]

sum_baseline1<-summary(trans1_streams)
sum_baseline2<-summary(trans2_streams)
sum_baseline3<-summary(trans3_streams)


fit_streams_new <- run_streams2(
  data              = panel_data,
  cov_vector        = cov_vector,
  python            = Sys.which("python"),
  pu_args           = list(verbose = TRUE),
  features_prop_add = NULL
)

class(fit_streams_new)

trans1_streams_new <- fit_streams_new[[1L]]
trans2_streams_new <- fit_streams_new[[2L]]
trans3_streams_new <- fit_streams_new[[3L]]

sum1_new<-summary(trans1_streams_new)
sum2_new<-summary(trans2_streams_new)
sum3_new<-summary(trans2_streams_new)





