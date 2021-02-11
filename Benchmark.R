library(lme4)
# Reading and combining data
X1 <- read.csv('/Users/liwentao/Documents/Research/GLMM/X1.csv', sep = ",", header = F)
X2 <- read.csv('/Users/liwentao/Documents/Research/GLMM/X2.csv', sep = ",", header = F)
y1 <- read.csv('/Users/liwentao/Documents/Research/GLMM/y1.csv', sep = ",", header = F)
y2 <- read.csv('/Users/liwentao/Documents/Research/GLMM/y2.csv', sep = ",", header = F)
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
