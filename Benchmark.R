library(lme4)
# Reading and combining data
X1 <- read.csv('/Users/wli17/Documents/GLMM/GLMM/X1.csv', sep = ",", header = F)
X2 <- read.csv('/Users/wli17/Documents/GLMM/GLMM/X2.csv', sep = ",", header = F)
y1 <- read.csv('/Users/wli17/Documents/GLMM/GLMM/y1.csv', sep = ",", header = F)
y2 <- read.csv('/Users/wli17/Documents/GLMM/GLMM/y2.csv', sep = ",", header = F)
X <- rbind(X1, X2)
y <- rbind(y1, y2)
names(y) <- 'outcome'
site <- rbind(y1*0+1, y2*0)
names(site) <- 'site'
dat <- cbind(y, X, site)
# glmer with 5 aGH fit
summary(glmer(outcome ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 + (1 | site) - 1, data = dat, family = binomial, nAGQ = 10))
head(y)


##### glmmML in the paper
library(glmmML)
glmmML(outcome ~ V1 + V2 + V3 + V4 + V5 + V6 + V7 + V8 + V9 + V10 -1, data = dat, cluster = site, n.points=1)
View(glmmML)
View(glmmML.fit)
View(glm.fit)


id <- factor(rep(1:20, rep(5, 20)))
y <- rbinom(100, prob = rep(runif(20), rep(5, 20)), size = 1)
x <- rnorm(100)
dat <- data.frame(y = y, x = x, id = id)
glmmML(y ~ x, data = dat, cluster = id)
glm(y ~ x, data = dat, family = 'binomial')


#### test in Penn data ####
dat2 <- read.csv('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_1/2Sites_500PatientsEachSite_SmallVar1_Dataset3.csv', 
                 sep = ',', header = T)
dat2$Site_ID

fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = dat2, cluster = Site_ID, n.points=1)
# summary(glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = dat2, family = binomial, nAGQ = 1))

Truth <- c(-1.5,0.1,-0.5,-0.3,0.4,-0.2,-0.25,0.35,-0.1,0.5)
# Setting 1
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_1')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_1/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_1_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}



# Setting 2
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_2')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_2/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_2_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}



# Setting 3
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_3')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_3/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_3_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}


# Setting 4
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_4')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_4/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_4_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}


# Setting 5
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_5')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_5/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_5_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}



# Setting 6
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_6')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_6/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_6_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}



# Setting 7
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_7')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_7/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_7_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}


# Setting 8
file_names <- list.files('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_8')
for (i in 1:length(file_names)){
  df <- read.csv(paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Setting_8/', file_names[i], sep = ''), sep = ',', header = T)
  # fit <- glmmML(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 - 1, data = df, cluster = Site_ID - 1, n.points=1)
  fit <- glmer(y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + (1 | Site_ID) - 1, data = df, family = binomial, nAGQ = 1)
  tab <- summary(fit)$coefficients
  CI_025  = tab[,1] - 1.959964 * tab[,2]
  CI_975  = tab[,1] + 1.959964 * tab[,2]
  out <- cbind(Truth, tab, CI_025, CI_975)
  colnames(out) <- c('Truth', 'Coef', 'Std.Err', 'z', 'P-value', '[0.025', '0.975]')
  write.csv(out, paste('/Users/wli17/Documents/GLMM/Simulation_data_GLMM/Result_R/Setting_8_', 
                       unlist(strsplit(file_names[i], split = 'Dataset'))[2], sep = ''))
}











