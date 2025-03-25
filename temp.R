library(data.table)
library(rsvd)
library(dplyr)

bb <- as.integer(readLines("temp.memb.gz")) + 1
.prop <- fread("temp.prop.gz")
kk <- apply(t(.prop), 2, which.max)

.beta.true <- as.matrix(fread("temp.dict.gz"))
.delta.true <- as.matrix(fread("temp.ln_batch.gz"))
.delta <- as.matrix(fread("temp3.delta.log_mean.gz"))
cor(.delta, .delta.true)

## .dt <- fread("temp3.proj.gz")
## .svd <- rsvd::rsvd(as.matrix(.dt), k=5)

## par(mfrow=c(2,2))
## plot(.svd$u[,1:2], col=bb)
## plot(.svd$u[,2:3], col=bb)
## plot(.svd$u[,3:4], col=bb)
## plot(.svd$u[,4:5], col=bb)

## par(mfrow=c(2,2))
## plot(.svd$u[,1:2], col=kk)
## plot(.svd$u[,2:3], col=kk)
## plot(.svd$u[,3:4], col=kk)
## plot(.svd$u[,4:5], col=kk)

.llik <- readLines("temp3.llik.gz")
par(mfrow=c(1,1))
plot(as.numeric(.llik), type="l")

.beta <- exp(as.matrix(fread("temp3.logit_dict.gz")))
pheatmap::pheatmap(cor(.beta, .beta.true, method="spearman"))


.dt <- fread("temp3.logit_latent.gz")
.mat <- as.matrix(.dt)
## .valid <- which(apply(t(.mat), 2, function(x) sum(is.na(x))) == 0)
## .invalid <- which(apply(t(.mat), 2, function(x) sum(is.na(x))) > 0)
## .mat <- .mat[.valid, ]
## .bb <- bb[.valid]
## .kk <- kk[.valid]

pheatmap::pheatmap(exp(.mat[order(kk), ]), cluster_rows = F)

pheatmap::pheatmap(exp(.mat[order(bb), ]), cluster_rows = F)



zz <- apply(exp(.mat) %*% matrix(rnorm(100), nrow=10, ncol=10), 2, scale)

.umap <- uwot::umap(zz, n_neighbors=30, n_components=2, n_threads=10, verbose = T, spread = 5, min_dist = 0)

par(mfrow=c(1,2))
plot(.umap, col = bb, cex=.3, pch=19)
plot(.umap, col = kk, cex=.3, pch=19)

.svd <- rsvd::rsvd(.mat, k=5)

par(mfrow=c(2,2))
plot(.svd$u[,1:2], col=bb)
plot(.svd$u[,2:3], col=bb)
plot(.svd$u[,3:4], col=bb)
plot(.svd$u[,4:5], col=bb)

par(mfrow=c(2,2))
plot(.svd$u[,1:2], col=kk, cex=.3)
plot(.svd$u[,2:3], col=kk, cex=.3)
plot(.svd$u[,3:4], col=kk, cex=.3)
plot(.svd$u[,4:5], col=kk, cex=.3)

