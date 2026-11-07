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

liquor_cv_fit <- liquor_cv |>
  model(snaive = SNAIVE(sales),
        ets    = ETS(log(sales)),
        arima  = ARIMA(log(sales)))

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








