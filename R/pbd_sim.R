#' Function to simulate the protracted speciation process
#'
#' Simulating the protracted speciation process using the Doob-Gillespie
#' algorithm. This function differs from pbd_sim_cpp that 1) it does not
#' require that the speciation-initiation rate is the same for good and
#' incipient species, and 2) that it simulates the exact protracted speciation
#' process, and not the approximation made by the coalescent point process.
#' This function provides also the conversion to the approximation as output.
#'
#'
#' @param pars Vector of parameters: \cr \cr \code{pars[1]} corresponds to b_1,
#' the speciation-initiation rate of good species \cr \code{pars[2]}
#' corresponds to la_1, the speciation-completion rate \cr \code{pars[3]}
#' corresponds to b_2, the speciation-initiation rate of incipient species \cr
#' \code{pars[4]} corresponds to mu_1, the extinction rate of good species \cr
#' \code{pars[5]} corresponds to mu_2, the extinction rate of incipient species
#' \cr
#' @param age Sets the age for the simulation
#' @param soc Sets whether this age is the stem (1) or crown (2) age
#' @param plotit Sets whether the various trees produced by the function should
#' be plotted or not
#' @param limitsize Sets a maximum to the number of incipient + good species
#' that are created during the simulation; if exceeded, the simulation is
#' aborted and removed.
#' @param add_shortest_and_longest Gives the output of the new samplemethods
#' 'shortest' and 'longest'.
#' @return \item{out}{ A list with the following elements: \cr \cr \code{tree}
#' is the tree of extant species in phylo format \cr \code{stree_random} is a
#' tree with one random sample per species in phylo format \cr
#' \code{stree_oldest} is a tree with the oldest sample per species in phylo
#' format \cr \code{stree_youngest} is a tree with the youngest sample per
#' species in phylo format \cr \code{L} is a matrix of all events in the
#' simulation where \cr - the first column is the incipient-level label of a
#' species \cr - the second column is the incipient-level label of the parent
#' of the species \cr - the third column is the time at which a species is born
#' as incipient species\cr - the fourth column is the time of
#' speciation-completion of the species \cr If the fourth element equals -1,
#' then the species is still incipient.  - the fifth column is the time of
#' extinction of the species \cr If the fifth element equals -1, then the
#' species is still extant.  - The sixth column is the species-level label of
#' the species \cr \code{sL_random} is a matrix like L but for
#' \code{stree_random} \cr \code{sL_oldest} is a matrix like L but for
#' \code{stree_oldest} \cr \code{sL_youngest} is a matrix like L but for
#' \code{stree_youngest} \cr \code{igtree.extinct} is the tree in simmap format
#' with incipient and good flags and including extinct species \cr
#' \code{igtree.extant} is the tree in simmap format with incipient and good
#' flags without extinct species \cr \code{recontree} is the reconstructed tree
#' in phylo format, reconstructed using the approximation in Lambert et al.
#' 2014 \cr \code{reconL} is the matrix corresponding to \code{recontree} \cr
#' \code{L0} is a matrix where the crown age is at 0; for internal use only \cr
#' }
#' @author Rampal S. Etienne
#' @seealso \code{\link{pbd_sim_cpp}}
#' @keywords models
#' @examples
#'  pbd_sim(c(0.2,1,0.2,0.1,0.1),15)
#' @export pbd_sim
pbd_sim <- function(pars, age, soc = 2, limitsize = Inf)
{
  la1 <- pars[1]
  la2 <- pars[2]
  la3 <- pars[3]
  mu1 <- pars[4]
  mu2 <- pars[5]

  i <- 1
  while (i <= soc)
  {
    t <- 0
    if (i == 1)
    {
      id1 <- 0
      id <- id1 + 1
      Sid1 <- 0
      Sid <- 1
      sg <- id
      si <- NULL
      L <- t(as.matrix(c(id, 0, -1e-10, t, -1, 1)))
    }
    if (i == 2)
    {
      id <- id1 + 1
      Sid <- Sid1
      sg <- NULL
      si <- -id
      L <- t(as.matrix(c(id, 1, t, -1, -1, 1)))
    }

    Ng <- length(sg)
    Ni <- length(si)
    probs <- c(la1 * Ng, mu1 * Ng, la2 * Ni, la3 * Ni, mu2 * Ni)
    denom <- sum(probs)
    probs <- probs / denom
    t <- t - log(stats::runif(1)) / denom

    while (t <= age)
    {
      event <- DDD::sample2(1:5, size = 1, prob = probs)
      if (event == 1)
      {
        parent <- as.numeric(DDD::sample2(sg, 1))
        id <- id + 1
        L <- rbind(L, c(id, parent, t, -1, -1, L[abs(parent) - id1, 6]))
        si <- c(si, -id)
        Ni <- Ni + 1
      }
      if (event == 2)
      {
        todie <- as.numeric(DDD::sample2(sg, 1))
        L[todie - id1, 5] <- t
        sg <- sg[-which(sg == todie)]
        Ng <- Ng - 1
      }
      if (event == 3)
      {
        tocomplete <- abs(as.numeric(DDD::sample2(si, 1)))
        L[tocomplete - id1, 4] <- t
        Sid <- Sid + 1
        L[tocomplete - id1, 6] <- Sid
        sg <- c(sg, tocomplete)
        si <- si[-which(abs(si) == tocomplete)]
        Ng <- Ng + 1
        Ni <- Ni - 1
      }
      if (event == 4)
      {
        parent <- as.numeric(DDD::sample2(si, 1))
        id <- id + 1
        L <- rbind(L, c(id, parent, t, -1, -1, L[abs(parent) - id1, 6]))
        si <- c(si, -id)
        Ni <- Ni + 1
      }
      if (event == 5)
      {
        todie <- abs(as.numeric(DDD::sample2(si, 1)))
        L[todie - id1, 5] <- t
        si <- si[-which(abs(si) == todie)]
        Ni <- Ni - 1
      }
      if (Ng + Ni > limitsize)
      {
        Ni <- 0
        Ng <- 0
      }
      probs <- c(la1 * Ng, mu1 * Ng, la2 * Ni, la3 * Ni, mu2 * Ni)
      denom <- sum(probs)
      probs <- probs / denom
      t <- t - log(stats::runif(1)) / denom
    }
    if (i == 1)
    {
      if ((Ng + Ni) > 0)
      {
        i <- i + 1
        L1 <- L
        id1 <- id
        Sid1 <- Sid
      }
    } else {
      if (i == 2)
      {
        if (checkgood(L, si, sg) == 1)
        {
          i <- i + 1
          L2 <- L
        }
      } }
  }
  L <- L1
  if (soc == 2)
  {
    L <- rbind(L1, L2)
  }
  absL <- L
  absL[, 2] <- abs(L[, 2])
  tree <- ape::as.phylo(ape::read.tree(text = detphy(absL, age)))
  pars <- c(la1, la2, la3, mu1, mu2)
  brts <- treestats::branching_times(tree)

  Ltreeslist <- list(tes = tree, pars = pars, age = age, brts = brts)

  return(Ltreeslist)
}


checkgood <- function(L, si, sg)
{
  j <- 1
  found <- 0
  if (length(sg) > 0)
  {
    found <- 1
  } else {
    if (length(si) == 0) { found <- 0 } else {
      while (found == 0 & j <= length(si))
      {
        rowinc <- which(L[, 1] == abs(si[j]))
        parent <- L[rowinc, 2]
        birth <- L[rowinc, 3]
        while (found == 0 & parent > 1)
        {
          rowpar <- which(L[, 1] == parent)
          if (L[rowpar, 4] > -1 & L[rowpar, 4] < birth)
            #if(L[parent - id1,4] > -1 & L[parent - id1,4] < birth)
          {
            found <- 1
          } else
          {
            parent <- L[rowpar, 2]
            birth <- L[rowpar, 3]
          }
        }
        j <- j + 1
      } } }
  invisible(found)
}


detphy <- function(L, age, ig = F, dropextinct = T)
{
  dimL <- dim(L)
  if ((dimL[1] == 1))
  {
    linlist <- paste0("(S-1-1-1:", age, ");")
  } else {
    L <- L[order(L[, 1]), 1:6]
    if (dropextinct == T)
    {
      sall <- which(L[, 5] == -1)
      tend <- age
    } else {
      sall <- which(L[, 5] >= -1)
      tend <- (L[, 5] == -1) * age + (L[, 5] > -1) * L[, 5]
    }

    linlist <- matrix(0, nrow = 1, ncol = 8)
    if (length(sall) == 1)
    {
      linlist[1,] <- c(L[sall,], paste0("S", paste0(L[sall, 6], L[sall, 6], L[sall, 1])), tend)
    } else {
      linlist <- cbind(L[sall,], paste0("S", paste0(L[sall, 6], L[sall, 6], L[sall, 1])), tend)
    }
    done <- 0
    while (done == 0)
    {
      j <- which.max(linlist[, 3])
      parent <- as.numeric(linlist[j, 2])
      parentj <- which(linlist[, 1] == parent)
      parentinlist <- length(parentj)

      if (parentinlist == 1)
      {
        startedge <- as.numeric(linlist[j, 3])
        comptime <- as.numeric(linlist[j, 4])
        endedge <- as.numeric(linlist[j, 8])
        comptimeparent <- as.numeric(linlist[parentj, 4])
        endedgeparent <- as.numeric(linlist[parentj, 8])
        if (ig == FALSE)
        {
          spec1 <- paste0(linlist[parentj, 7], ":", endedgeparent - startedge)
          spec2 <- paste0(linlist[j, 7], ":", endedge - startedge)
        } else {
          if (comptimeparent == -1 | comptimeparent > endedgeparent)
          {
            comptimeparent <- endedgeparent
          }
          itimeparent <- max(0, comptimeparent - startedge)
          gtimeparent <- endedgeparent - max(comptimeparent, startedge)
          if (itimeparent == 0)
          {
            spec1 <- paste0(linlist[parentj, 7], ":{g,", gtimeparent, ":i,", itimeparent, ":g,0}")
          } else {
            spec1 <- paste0(linlist[parentj, 7], ":{g,", gtimeparent, ":i,", itimeparent, "}")
          }
          if (comptime == -1 | comptime > endedge)
          {
            comptime <- endedge
          }
          itime <- comptime - startedge
          gtime <- endedge - comptime
          if (itimeparent == 0)
          {
            spec2 <- paste0(linlist[j, 7], ":{g,", gtime, ":i,", itime, ":g,0}")
          } else {
            spec2 <- paste0(linlist[j, 7], ":{g,", gtime, ":i,", itime, "}")
          }
        }
        linlist[parentj, 7] <- paste0("(", spec1, ",", spec2, ")")
        linlist[parentj, 8] <- linlist[j, 3]
        linlist <- linlist[-j,]
      } else {
        if (as.numeric(parent) != 0)
        {
          parentj2 <- which(L[, 1] == as.numeric(parent))
          comptimeparent2 <- L[parentj2, 4]

          #print(paste('parentj2 is ',parentj2))
          #print(L)
          #print(paste('comptimeparent is ',comptimeparent2))

          if (comptimeparent2 > -1 & (comptimeparent2 < as.numeric(linlist[j, 3]) | parentj2 <= -1))
          {
            linlist[j, 4] <- L[parentj2, 4]
          }
          linlist[j, c(1:3, 5)] <- L[parentj2, c(1:3, 5)]
        }
      }
      if (is.null(nrow(linlist)))
      {
        done <- 1
        if (ig == FALSE)
        {
          linlist[7] <- paste0(linlist[7], ":", abs(as.numeric(linlist[3])), ";")
        } else {
          linlist[7] <- paste0(linlist[7], ";")
        }
      } else {
        if (nrow(linlist) == 1)
        {
          done <- 1
          if (ig == FALSE)
          {
            linlist[7] <- paste0("(", linlist[7], ":", age, ");")
          } else {
            linlist[7] <- paste0(linlist[7], ";")
          }
        }
      }
    }
  }
  return(linlist[7])
}