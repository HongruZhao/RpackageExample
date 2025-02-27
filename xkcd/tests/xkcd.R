library(xkcd)

dxkcd(1/sqrt(2*pi)) == 0
dxkcd(0) == Inf
pxkcd(0) == 0
pxkcd(0, log.p = T) == -Inf
pxkcd(1/sqrt(2*pi)) == 1
pxkcd(1/sqrt(2*pi), log.p = T) == 0

## Using "log = TRUE" for an extended range :
par(mfrow = c(2,1))
plot(function(x) dxkcd(x, log = TRUE), 0, 1/sqrt(2*pi),
     main = "log { Xkcd density }")
curve(log(dxkcd(x)), add = TRUE, col = "red", lwd = 2)
mtext("dxkcd(x, log=TRUE)", adj = 0)
mtext("log(dxkcd(x))", col = "red", adj = 1)

plot(function(x) pxkcd(x, log.p = TRUE), 0, 1/sqrt(2*pi),
     main = "log { Xkcd Cumulative }")
curve(log(pxkcd(x)), add = TRUE, col = "red", lwd = 2)
mtext("pxkcd(x, log=TRUE)", adj = 0)
mtext("log(pxkcd(x))", col = "red", adj = 1)