library(tidyverse)
library(fpp3)
library(BVAR)

# 2.1: Load and Plot Liquor Data ----
liquor_raw <- read_csv(
  "data/MRTSSM4453USN.csv",
  na = ".",
  col_types = cols(
    observation_date = col_date(format = "%Y-%m-%d"),
    MRTSSM4453USN = col_double()
  )
)

liquor <- liquor_raw |>
  transmute(Month = yearmonth(observation_date), sales = MRTSSM4453USN) |>
  as_tsibble(index = Month) |>
  filter(Month <= yearmonth("2019 Dec"))

has_gaps(liquor)

liquor |> 
  autoplot(sales) +
  labs(x = "Month", 
       y = "Liquor Sales (Millions of Dollars)", 
       title = "US Retail Liquor Sales by Month (1992-2019)")

gg_season(liquor, y = log(sales)) +
          labs(x = "Month", 
               y = "Log Sales", 
               title = "Logged US Retail Liquor Store Sales by Month (1992-2019)")

# There is a clear upward sloping trend and seasonal pattern, with  sales increasing 
# more or less linearly over time, and a big spike every year in December. Sales are 
# typically lowest in Jan/Fed and highest in December, and the size of the seasonal 
# swings grows with the level, so this series is a good candidate for a log 
# transformation. 







# 2.2: Fitting ARIMA and ETS ----
liquor_fit <- liquor |>
  model(arima = ARIMA(log(sales)),
        ets   = ETS(log(sales)))

liquor_fit |> select(arima) |> report()
liquor_fit |> select(ets) |> report()

# An ARIMA(1, 0, 2)(2, 1, 1)[12] model was chosen, meaning that there is one seasonal
# difference applied to the series. An ETS(A, A, A), an additive Holt-Winters model
# was chosen. The 3 A's mean that the model uses additive errors, trends, and 
# seasonal components. 

liquor_fit |> select(arima) |> gg_tsresiduals()

liquor_fit |>
  select(arima) |>
  augment() |>
  features(.innov, ljung_box, lag = 24, dof = 6)

# Yes, the residuals look like white noise, and the L-B test confirms this. 







# 2.3: The Horse Race ----
liquor_cv <- liquor |>
  stretch_tsibble(.init = 120, .step = 1) |>
  filter(.id < max(.id))

liquor_cv_fit <- readRDS("results/liquor_cv_fit.rds")

saveRDS(liquor_cv_fit, "results/liquor_cv_fit.rds")

# There will be 216 different forecast origins considered. We have 28 years of 
# monthly data, and 10 of those years make the first training sample and the first
# forecast origin. We then have 18 years of monthly data -> 18x12 = 216 origins, 
# plus the first forecast origin minus the dropped December 2019 origin. 

liquor_cv_fc <- liquor_cv_fit |>
  forecast(h = 1)

liquor_cv_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

# January 2002 is the first forecase and December 2019 is the last. 
# Model  Type  RMSE   MAE
# 
# arima  Test  81.4   65.7
# ets    Test  94.4   72.3
# snaive Test  161.   137. 
#
# The ARIMA model forecast the most accurately, having both the loweset MAE and 
# RMSE. 

# 2.4: DM Test ----
liquor_errs <- liquor_cv_fc |>
  as_tibble() |>
  left_join(as_tibble(liquor) |> select(Month, actual = sales),
            by = "Month") |>
  mutate(e = actual - .mean) |>
  select(Month, .model, e) |>
  pivot_wider(names_from = .model, values_from = e)

forecast::dm.test(liquor_errs$arima, liquor_errs$ets, h = 1, power = 2)
forecast::dm.test(liquor_errs$arima, liquor_errs$snaive, h = 1, power = 2)

# ARIMA vs. ETS: DM of -2.9062, p-value of 0.004042
# ARIMA vs. SRW: DM of -10.755, p-value of 2.2e-16

# The Diebold-Mariano tests confirm that the differences in forecast errors were 
# significant. ARIMA is better than both at the 1% significance level. If I were
# the liquor store, I would want to use the ARIMA model. 









# 3: Forecast Combination ----
liquor_combo_fc <- liquor_cv_fit |>
  mutate(combo = (arima + ets) / 2) |>
  forecast(h = 1)

liquor_combo_fc |>
  accuracy(liquor, measures = list(RMSE = RMSE, MAE = MAE))

# Model  type  RMSE  MAE
#
# arima  Test  81.4  65.7
# combo  Test  85.2  66.8
# ets    Test  94.4  72.3
# snaive Test  161.  137. 
#
# Combo does better than the SRW and ETS models, but worse than the ARIMA model. 

cor(liquor_errs$arima, liquor_errs$ets)

# The combination does not beat the ARIMA model because the ARIMA and ETS 
# errors are highly correlated, and because the ETS is statistically 
# significantly less accurate than the ARIMA. The correlation coefficient of
# 0.878 is quite high, meaning that the models missed many of the same forecasts, 
# and the combined models errors are basically an average of the ARIMA's errors
# + the errors of the less accurate ETS.






# 4.1: Loading Data ----
qd_raw <- read_csv("data/fredqd_2026-09.csv", show_col_types = FALSE) |>
  slice(-(1:2))

growth <- function(x) 100 * difference(log(x))

qd <- qd_raw |>
  mutate(date = as.Date(sasdate, "%m/%d/%Y")) |>
  transmute(date,
            gdp    = growth(GDPC1),     # real GDP
            cons   = growth(PCECC96),   # real consumption
            inv    = growth(GPDIC1),    # real investment
            emp    = growth(PAYEMS),    # payroll employment
            infl   = growth(GDPCTPI),   # GDP price index
            unrate = difference(UNRATE),     # unemployment rate
            ffr    = difference(FEDFUNDS),   # federal funds rate
            gs10   = difference(GS10)) |>    # 10-year Treasury yield
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

y <- qd |> select(-date) |> as.matrix()

nrow(y)
head(y)

N <- nrow(y)
origins <- 120:(N - 1)
actual <- y[121:N, ]

# We will produce 116 forecasts for each variable (4 Quarterly forecasts for each of 
# the 29 years from 1990 to 2019). The first forecase will be for Q1 of 1990, and the 
# last will be for Q4 of 2019. 

# 4.2: OLS VAR's ----
var_ols_fc <- function(z, p) {
  e <- embed(z, p + 1)            # columns: y_t, then its lags 1 to p
  M <- ncol(z)                    # number of variables
  B <- coef(lm(e[, 1:M] ~ e[, -(1:M)]))
  x <- c(1, t(z[nrow(z):(nrow(z) - p + 1), ]))   # constant, latest p lags
  x %*% B
}

fc_var1 <- NULL
fc_var4 <- NULL
for (T in origins) {
  fc_var1 <- rbind(fc_var1, var_ols_fc(y[1:T, ], 1))
  fc_var4 <- rbind(fc_var4, var_ols_fc(y[1:T, ], 4))
}

rmse_by_var <- function(f) sqrt(colMeans((actual - f)^2))

rmse_by_var(fc_var1)
rmse_by_var(fc_var4)

# The VAR(1) forecasts more accurately for 7/8 of the variables. Since VAR(1) is 
# the special case of VAR(4) where the coeffficients on lags 2, 3, and 4 are equal
# to 0, it will fit as least as good as the VAR(1) in sample, and this might mean 
# that the VAR(4) is overfitting the training data. The increased estimation error 
# created by estimating 4 lags outweighs any benefits of potentially decreased 
# approximation error. 










# 4.3: BVAR ----
bvar_cv <- function(lambda) {
  pr <- bv_priors(hyper = "lambda",
                  mn = bv_mn(lambda = bv_lambda(mode = lambda, sd = 0.4,
                                                min = 0.999 * lambda,
                                                max = 1.001 * lambda),
                             b = 0))
  fc <- NULL
  for (T in origins) {
    fit_T <- bvar(y[1:T, ], lags = 4, priors = pr,
                  n_draw = 2000, n_burn = 1000, verbose = FALSE)
    x <- c(1, t(y[T:(T - 3), ]))         # constant and 4 lags of each variable
    fc <- rbind(fc, x %*% coef(fit_T))   # forecast of y_{T+1}
  }
  fc
}

set.seed(422)
fc_bvar_03 <- bvar_cv(0.3)
saveRDS(fc_bvar_03, "results/fc_bvar_03.rds")

set.seed(422)
fc_bvar_005 <- bvar_cv(0.05)

var_results <- tibble(
  Variable = colnames(y),
  `VAR(1)` = rmse_by_var(fc_var1),
  `VAR(4)` = rmse_by_var(fc_var4),
  `BVAR(4), lambda = 0.3` = rmse_by_var(fc_bvar_03),
  `BVAR(4), lambda = 0.05` = rmse_by_var(fc_bvar_005)
)

saveRDS(var_results, "results/var_results.rds")

readRDS("results/var_results.rds") |> knitr::kable(digits = 3)

# The BVAR(4) with lambda = 0.3 does better than the VAR(4) for all of the variables. 
# The difference is explained by the shrinkage performed in the BVAR, which is reducing
# our estimation error. 
# 
# It does worse than the VAR(1) for 5/8 of the variables, which could be an indicator
# that the additional lags are introducing more estimation error than useful 
# information.
# 
# When lamba = 0.05, it performs worse for everything except the ffr variable, which
# tells us that we should not be shrinnking this aggressively. I think that saying
# that the winners RMSE is a fair estimate of future forecast accuracy is too
# optimistic, because it could have just gotten lucky in this training sample. 
# We tested four different models across 8 variables, so there is a good chance some 
# of the low RMSE's are just due to chance.

e_var4 <- actual[, "gdp"] - fc_var4[, 1]
e_bvar <- actual[, "gdp"] - fc_bvar_03[, 1]

v2e_var4 <- actual[, "inv"] - fc_var4[, 3]
v2e_bvar <- actual[, "inv"] - fc_bvar_03[, 3]

forecast::dm.test(e_var4, e_bvar, h = 1, power = 2)
forecast::dm.test(v2e_var4, v2e_bvar, h = 1, power = 2)

# Based on the DM tests, we reject the null for GDP, and fail to reject the null for 
# the INV variable. This means that the accuracy gap is real for GDP, but we cannot 
# distinguish between the two models for INV. This reinforces how I felt about the 
# results for 4.3b, as it shows how one of the cases where it appeared the VAR(1) 
# was more accurate did now hold up through the DM test. 



# 5.1 Data ----
md_raw <- read_csv("data/fredmd_2026-09.csv", show_col_types = FALSE)
tcode <- unlist(md_raw[1, -1])
md <- md_raw[-1, ] |> mutate(date = as.Date(sasdate, "%m/%d/%Y"))

transform_series <- function(x, code) {
  d <- function(v) c(NA, diff(v))
  switch(as.character(code),
         "1" = x,                      # level
         "2" = d(x),                   # first difference
         "3" = d(d(x)),                # second difference
         "4" = log(x),                 # log
         "5" = 100 * d(log(x)),        # growth rate
         "6" = 100 * d(d(log(x))),     # change in the growth rate
         "7" = d(x / dplyr::lag(x) - 1))
}

fm <- map2_dfc(md[names(tcode)], tcode, transform_series) |>
  mutate(date = md$date, .before = 1) |>
  filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))

fm <- fm[, colSums(is.na(fm)) == 0]

dim(fm)

# INDPRO and PAYEMS are transformed into their growth rates, CPIAUCSL is transformed 
# into the change in the growth rate, and FEDFUNDS is transformed with a first 
# difference. The Federal Funds rate is transformed differently because it is a rate 
# that sits around a consistent range, and it is more useful to look at the change 
# in that rate rather than the growth rate of that rate. The Industrial Production 
# Index is transformed using the growth rate because it is easier to interpret the 
# growth rate of industrial production, rather than the difference in the index, 
# which is hard to contextualize because of the units. 









# 5.2: Factors ----
Z <- fm |> select(-date) |> as.matrix()

pc <- prcomp(scale(Z))

summary(pc)$importance[, 1:5]

# The first 3 principal components explain 29.453% of the variation. 

F1 <- pc$x[, 1] * sign(cor(pc$x[, 1], fm$INDPRO))

tibble(date = fm$date, F1 = F1) |>
  ggplot(aes(date, F1)) +
  geom_line() +
  labs(title = "First Factor from FRED-MD", x = NULL, y = NULL) +
  theme_minimal()

sort(abs(cor(pc$x[, 1], Z))[1, ], decreasing = TRUE)[1:8]
sort(abs(cor(pc$x[, 2], Z))[1, ], decreasing = TRUE)[1:8]

# The first factor captures the production side of the economy. It is highly correlated 
# with multiple series tracking employment and industrial production. 
# 
# The second factor captures the consumption side of the economy. It is highly 
# correlated with multiple series tracking consumption via the CPI. 




# 5.3: Horse Race ----
ip <- fm$INDPRO
n <- nrow(Z)
init <- which(fm$date == as.Date("1989-12-01"))

e_ar <- NULL
e_fa <- NULL
for (T in init:(n - 1)) {
  F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:3]         # factors using data through T
  d <- data.frame(y_next = ip[2:T], y = ip[1:(T - 1)], F_T[1:(T - 1), ])
  new <- data.frame(y = ip[T], t(F_T[T, ]))
  e_ar <- c(e_ar, ip[T + 1] - predict(lm(y_next ~ y, d), new))
  e_fa <- c(e_fa, ip[T + 1] - predict(lm(y_next ~ ., d), new))
}

c(RMSE_AR = sqrt(mean(e_ar^2)), RMSE_FA = sqrt(mean(e_fa^2)))

# Numbers are the same as we got in class. If we first estimated the factors using 
# the whole sample and then used them in the horse race, this would be cheating because
# we are using data that we would not have actually known at the time, aka creating 
# data leakage. 

# 5.4: Direct Forecast of Inflation ----
direct_race <- function(series, h = 12, r = 3) {
  lv <- md |>
    mutate(log_x = log(.data[[series]]),
           g = 100 * (log_x - lag(log_x))) |>
    filter(date >= as.Date("1960-03-01"), date <= as.Date("2019-12-01"))
  yh <- 100 * (lead(lv$log_x, h) - lv$log_x)      # growth over the next h months
  g  <- lv$g                                      # growth over the latest month
  
  e_ar <- NULL
  e_fa <- NULL
  for (T in init:(n - h)) {
    F_T <- prcomp(scale(Z[1:T, ]))$x[, 1:r, drop = FALSE]
    i <- 1:(T - h)
    d <- data.frame(yh = yh[i], g = g[i], F_T[i, , drop = FALSE])
    new <- data.frame(g = g[T], F_T[T, , drop = FALSE])
    e_ar <- c(e_ar, yh[T] - predict(lm(yh ~ g, d), new))
    e_fa <- c(e_fa, yh[T] - predict(lm(yh ~ ., d), new))
  }
  tibble(series, forecasts = length(e_ar),
         RMSE_AR = sqrt(mean(e_ar^2)), RMSE_FA = sqrt(mean(e_fa^2)),
         MAE_AR = mean(abs(e_ar)), MAE_FA = mean(abs(e_fa)),
         DM_pvalue = forecast::dm.test(e_ar, e_fa, h = h, power = 2,
                                       varestimator = "bartlett")$p.value)
}

direct_race("CPIAUCSL")

# Yes, the factors improve the forecast, reducing RMSE by 11.64%. This difference is 
# statistically significant at any realistic significance level, with a p-value of 
# 0.0004, meaning that we can reject the null of equal accuracy. 
