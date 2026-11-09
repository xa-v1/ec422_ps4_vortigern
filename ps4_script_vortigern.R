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

# 4.2L OLS VAR's ----
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









