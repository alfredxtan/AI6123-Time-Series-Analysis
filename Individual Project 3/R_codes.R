library(quantmod) #getSymbols
library(zoo)
library(xts)
library(tseries) # adf.test
library(TSA) # kurtosis, skewness
library(forecast)
library(rugarch)
library(changepoint)



stockname = 'AAPL'
stock.data = getSymbols(stockname, from='2002-02-01', to='2017-02-01', src='yahoo', auto.assign = F) 
stock.data = na.omit(stock.data)
chartSeries(stock.data, theme = "white", name = stockname)
class(stock.data) 
a.c = stock.data[,4] #Close price
names(a.c) = 'Apple Stock Prices (2002-2017)'
head(a.c)


## ----- Examine data ----- ###
plot(a.c, main="Closing price of Apple stock", type = 'l')

acf(a.c, main="ACF of closing price")
pacf(a.c, main="PACF of closing price")

adf.test(a.c) # p-value = 0.4114 (non-stationary)
kpss.test(a.c)

a.c.monthly = to.monthly(stock.data)
freq = 12
adj.ts = ts(a.c.monthly, frequency = freq)
fit.stl = stl(adj.ts[,1], s.window = "periodic")
autoplot(fit.stl, main="STL Decomposition")

periodogram(a.c, main ='Periodogram for Apple stock')
periodogram(a.c)$freq ; periodogram(a.c)$spec


## ----- Split train/test set ----- ##
num_of_train <- (length(a.c) - 30) %>% print()
a.c.train <- head(a.c, num_of_train)
a.c.test <- tail(a.c, round(length(a.c) - num_of_train))
length(a.c); length(a.c.train); length(a.c.test)
plot(a.c.train, main="Training data")


## ----- ARIMA Modelling ----- ##
lambda = BoxCox.lambda(a.c.train)
lambda
a.c.bc = BoxCox(a.c.train, lambda)
plot(a.c.bc, main = paste("Transformed closing price for lambda = ",lambda))
acf(a.c.bc, main = "SACF of transformed closing price")
pacf(a.c.bc, main = "SPACF of transformed closing price")

#Check for stationarity of one-time-differenced time series
a.c.bc.diff1 = na.omit(diff(a.c.bc))
plot(a.c.bc.diff1, main="One-time differenced transformed closing price")
acf(a.c.bc.diff1, main = "SACF of one-time differenced transformed closing price")
pacf(a.c.bc.diff1, main = "SPACF of one-time differenced transformed closing price")
adf.test(a.c.bc.diff1)


#Performing diagnostic testing and model selection
fit.arima014 = arima(a.c.bc, order=c(0,1,4))
tsdiag(fit.arima014)
AIC(fit.arima014) 

fit.arima410 = arima(a.c.bc, order=c(4,1,0))
tsdiag(fit.arima410)
AIC(fit.arima410) 
#AIC values
#-16701.04 for (0,1,4)
#-16701.17 for (4,1,0)


fit.arima = arima(a.c.bc, order=c(4,1,0))
forecast = predict(fit.arima, n.ahead=length(a.c.test))

forecast_y = InvBoxCox(forecast$pred, lambda)
forecast_ub = InvBoxCox(forecast$pred + 1.96*forecast$se, lambda) # upper bound
forecast_lb = InvBoxCox(forecast$pred - 1.96*forecast$se, lambda) # lower bound

train.index = index(a.c.train)
test.index = index(a.c.test)

fitted.model = InvBoxCox(a.c.bc-fit.arima$residuals, lambda)

plot(train.index, a.c.train, ylim=c(0,40), main='Apple Stock Close Price', type='l')
lines(test.index, forecast_ub, type='l', col='blue')
lines(test.index, forecast_lb, type='l', col='blue')
lines(test.index, forecast_y, type='l', col='red')
lines(test.index, a.c.test, type='l', col = 'black')
legend("topleft", 
       c("Close Price Data ", 
         "Model Forecast",  
         "Forecast Upper & Lower Bounds"), 
       col=c('black', 'red', 'blue'), 
       lty=c(1,1,1,1))

#Calculate MSE
actual_values = a.c.test
predicted_values = forecast_y
squared_errors = (actual_values - predicted_values)^2
mse = mean(squared_errors)
cat("Mean squared error: ", mse)

## ----- AUTO.ARIMA Forecast ----- ##
fit.aa = auto.arima(a.c.train, lambda=lambda, d=1) %>% print() #AIC = -16708.19
tsdiag(fit.aa)
AIC(fit.aa) 
Box.test(fit.aa$residuals, type="Ljung-Box")
pred.aa = forecast(fit.aa, h=length(a.c.test))


plot(train.index, a.c.train, ylim=c(0,50), main='Apple Stock Close Price', type='l')
lines(test.index, pred.aa$lower[,2], type='l', col='blue')
lines(test.index, pred.aa$upper[,2], type='l', col='blue')
lines(test.index, pred.aa$mean, type='l', col='red')
lines(test.index, a.c.test, type='l', col = 'black')
legend("topleft", 
       c("Close Price Data ", 
         paste("Model Forecast:", pred.aa$method),  
         "Forecast Upper & Lower Bounds"), 
       col=c('black', 'red', 'blue'), 
       lty=c(1,1,1,1))

#Calculate MSE
actual_values = a.c.test
predicted_values = pred.aa$mean
squared_errors = (actual_values - predicted_values)^2
mse = mean(squared_errors)
cat("Mean squared error: ", mse)


## ----- Data transformation for GARCH ----- ##
lambda = 0 # log transform
percentage = 100 
a.r = diff(BoxCox(a.c, lambda))*percentage # returns 
a.r = a.r[!is.na(a.r)] 
length(a.c); length(a.r)
names(a.r) = 'Apple Daily Relative Return (2002-2017)'
plot(a.r, main=names(a.r))

# Plot timeplot and ACF PACF Plots of Abs and Sq
acf(a.r, main="ACF of Returns") # acf cut off after lag 4 
pacf(a.r, main = "PACF of Returns") #pacf cuts off after lag 4
adf.test(a.r) # p-value = 0.01

plot(abs(a.r), main ="Absolute Returns")
acf(abs(a.r), main="ACF of Absolute Returns ") #Decay slowly

plot(a.r^2, main ="Squared Returns")
acf(a.r^2, main = "ACF of Squared Returns") #Decay slowly


## ----- Plot QQ Plot ----- ##
qqnorm(a.r)
qqline(a.r, col = 2)
skewness(a.r); kurtosis(a.r) #-0.1901292; 5.44009

## ----- Split train/test for returns-----##
num_of_train <- (length(a.r) - 30) %>% print()
a.r.train <- head(a.r, num_of_train)
a.r.test <- tail(a.r, round(length(a.r) - num_of_train))
length(a.r); length(a.r.train); length(a.r.test)
plot(a.r.train, main = 'Training set for Returns')


## ----- EACF ----- ##
eacf(a.r.train^2)
eacf(abs(a.r.train))


## ----- GARCH ----- ##
#GARCH(1,1) and GARCH(2,2)

### GARCH 11 ###
garch.11=garch(a.r.train, order=c(1,1))
plot(residuals(garch.11), type='h', ylab='Standardized Residuals', main = 'GARCH(1,1)')
qqnorm(residuals(garch.11)); qqline(residuals(garch.11), col = 2)
kurtosis(na.omit(residuals(garch.11)))
summary(garch.11)

acf(na.omit(abs(residuals(garch.11))), main = "Absolute residuals")
acf(na.omit(residuals(garch.11)^2), main = 'Sqaured residuals')
gBox(garch.11,method='squared')

### GARCH22 ####
garch.22 = garch(a.r.train, order=c(2,2))
plot(residuals(garch.22), type='h', ylab='Standardized Residuals', main = 'GARCH(2,2)')
qqnorm(residuals(garch.22)); qqline(residuals(garch.22), col = 2)
kurtosis(na.omit(residuals(garch.22)))

acf(na.omit(abs(residuals(garch.22))), main = "Absolute residuals")
acf(na.omit(residuals(garch.22)^2), main = 'Sqaured residuals')
gBox(garch.22,method='squared')
summary(garch.22)


### GARCH model fitting ###
s = ugarchspec(mean.model = list(armaOrder = c(0,0)),
               variance.model = list(model = 'eGARCH', garchOrder=c(1,1)), 
               distribution.model = 'std')
fit.garch = ugarchfit(data=a.r.train, spec=s)
plot(fit.garch, which = 9)
setfixed(s) = as.list(coef(fit_garch))

# plot forcasted close price and mean
forecast.returns = ugarchpath(spec = s, m.sim=3, n.sim = length(a.c.test))
forecast.close = exp(apply(fitted(forecast.returns)/100, 2, 'cumsum')) + as.numeric(a.c.train[3745])
mean.close = rowMeans(forecast.close)
matplot(forecast.close, type='l')
lines(mean.close, lty = 4, col = "blue") 

#Simulations until criterion hit
count=0
max_value = 0
value = max(a.c.test)+rnorm(1,0,0.1))
print(value)
while (max_value < value){
  forecast.returns = ugarchpath(spec = s, m.sim = 1, n.sim = length(a.c.test))
  forecast.close = exp(apply(fitted(forecast.returns)/100, 2, "cumsum")) + as.numeric(a.c.train[3745])
  count = count + 1
  #max_value = max(forecast.close)
  min_value = min(forecast.close)
  if (count%%10 == 0){
  print(count)} 
}
print(paste("Value:", value))
print(paste("Final count: ",count))
matplot(forecast.close, type='l')

#Simulate one randomly
forecast.returns = ugarchpath(spec = s, m.sim=1, n.sim = length(a.c.test))
forecast.close = exp(apply(fitted(forecast.returns)/100, 2, 'cumsum')) + as.numeric(a.c.train[3745])


plot(train.index, a.c.train, ylim=c(0,50), xlab='Year', ylab='USD', main='Apple Stock Close Price', type='l')
lines(test.index, a.c.test, col = 'black')
lines(test.index, forecast.close, col = "red")
legend("topleft", 
       c("Ground truth closing price", 
         "Simulated Forecast"),
       col=c('black', 'red'), 
       lty=c(1,1,1,1))


#Calculate MSE
actual_values = a.c.test
predicted_values = forecast.close
squared_errors = (actual_values - predicted_values)^2
mse = mean(squared_errors)
cat("Mean squared error: ", mse)



