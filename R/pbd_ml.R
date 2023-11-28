#' Maximization of loglikelihood under protracted birth-death model of
#' diversification
#'
#' Likelihood maximization for protracted birth-death model of diversification
#'
#'
#' @param brts A set of branching times of a phylogeny, all positive
#' @param initparsopt The initial values of the parameters that must be
#' optimized:
#' \itemize{
#'   \item initparsopt[1] = b (= la_1 in ER2012) =
#'     speciation initiation rate
#'   \item initparsopt[2] = mu_1 (= mu_g in ER2012) =
#'     extinction rate of good species
#'   \item initparsopt[3] = la_1 (= la_2 in ER2012) =
#'     speciation completion rate
#'   \item initparsopt[4] = mu_2 (= mu_i in ER2012) =
#'     extinction rate of incipient species
#' }
#' @param idparsopt The ids of the parameters that must be optimized, e.g. 1:4
#' for all parameters.  The ids are defined as follows: \cr id == 1 corresponds
#' to b (speciation-initiation rate) \cr id == 2 corresponds to mu_1
#' (extinction rate of good species) \cr id == 3 corresponds to la_1
#' (speciation-completion rate) \cr id == 4 corresponds to mu_2 (extinction
#' rate of incipient species) \cr
#' @param idparsfix The ids of the parameters that should not be optimized,
#' e.g. c(2,4) if mu_1 and mu_2 should not be optimized, but only b and la_1.
#' In that case idparsopt must be c(1,3).
#' @param parsfix The values of the parameters that should not be optimized
#' @param exteq Sets whether incipient species have the same (1) or different
#' (0) extinction rate as good species. If exteq = 0, then idparsfix and
#' idparsopt should together have all parameters 1:4
#' @param parsfunc Specifies functions how the rates depend on time, default
#' functions are constant functions
#' @param missnumspec The number of species that are in the clade but missing
#' in the phylogeny
#' @param cond Conditioning: \cr cond == 0 : conditioning on stem or crown age
#' \cr cond == 1 : conditioning on stem or crown age and non-extinction of the
#' phylogeny \cr cond == 2 : conditioning on stem or crown age and number of
#' extant taxa \cr
#' @param btorph Sets whether the likelihood is for the branching times (0) or
#' the phylogeny (1)
#' @param soc Sets whether the first element of the branching times is the stem
#' (1) or the crown (2) age
#' @param methode Sets which method should be used in the ode-solver. Default
#' is 'lsoda'. See package deSolve for details.
#' @param n_low Sets the lower bound of the number of species on which
#' conditioning should be done when cond = 2. Set this to 0 when conditioning
#' should be done on precisely the number of species (default)
#' @param n_up Sets the upper bound of the number of species on which
#' conditioning should be done when cond = 2. Set this to 0 when conditioning
#' should be done on precisely the number of species (default)
#' @param tol Sets the tolerances in the optimization. Consists of: \cr reltolx
#' = relative tolerance of parameter values in optimization \cr reltolf =
#' relative tolerance of function value in optimization \cr abstolx = absolute
#' tolerance of parameter values in optimization
#' @param maxiter Sets the maximum number of iterations in the optimization
#' @param optimmethod Method used in optimization of the likelihood. Current
#' default is 'subplex'. Alternative is 'simplex' (default of previous
#' versions)
#' @param num_cycles Number of cycles of the optimization (default is 1).
#' @param verbose if TRUE, explanatory text will be shown
#' @return A data frame with the following components:\cr
#' \item{b}{ gives the maximum likelihood estimate of b}
#' \item{mu_1}{ gives the maximum likelihood estimate of mu_1}
#' \item{la_1}{ gives the maximum likelihood estimate of la_1}
#' \item{mu_2}{ gives the maximum likelihood estimate of mu_2}
#' \item{loglik}{ gives the maximum loglikelihood}
#' \item{df}{ gives the number of estimated parameters, i.e. degrees of feedom}
#' \item{conv}{ gives a message on convergence of optimization;
#' conv = 0 means convergence}
#' @author Rampal S. Etienne
#' @seealso \code{\link{pbd_loglik}}
#' @keywords models
#' @examples
#'
#' pbd_ML(1:10,initparsopt = c(4.640321,4.366528,0.030521), exteq = 1)
#'
#' @export pbd_ML
pbd_ML <- function(
  brts,
  initparsopt = c(0.2,0.1,1),
  idparsopt = 1:length(initparsopt),
  idparsfix = NULL,
  parsfix = NULL,
  exteq = 1,
  parsfunc = c(function(pars) {pars[1]},function(pars) {pars[2]},function(pars) {pars[3]},function(pars) {pars[4]}),
  missnumspec = 0,
  cond = 1,
  btorph = 1,
  soc = 2,
  methode = "lsoda",
  n_low = 0,
  n_up = 0,
  tol = c(1E-6, 1E-6, 1E-6),
  maxiter = 1000 * round((1.25)^length(idparsopt)),
  optimmethod = 'subplex',
  num_cycles = 1,
  verbose = TRUE)
{
  #options(warn=-1)
  brts <- sort(abs(as.numeric(brts)), decreasing = TRUE)
  if(is.numeric(brts) == FALSE)
  {
    cat("The branching times should be numeric.\n")
    out2 <- data.frame(b = -1, mu_1 = -1, lambda_1 = -1, mu_2 = -1, loglik = -1, df = -1, conv = -1)
  } else {
    if(exteq == 1){ idexteq <- 4 } else { idexteq <- NULL }
    idpars <- sort(c(idparsopt, idparsfix, idexteq))
    if((prod(idpars == (1:4)) != 1) || (length(initparsopt) != length(idparsopt)) || (length(parsfix) != length(idparsfix)))
    {
      cat("The arguments should be coherent.\n")
      out2 <- data.frame(b = -1, mu_1 = -1, lambda_1 = -1, mu_2 = -1, loglik = -1, df = -1, conv = -1)
    } else {
      namepars <- c("b", "mu_1", "lambda_1", "mu_2")
      if(length(namepars[idparsopt]) == 0) { optstr <- "nothing" } else { optstr <- namepars[idparsopt] }
      if (verbose) { cat("You are optimizing",optstr,"\n") }
      if(length(namepars[idparsfix]) == 0) { fixstr <- "nothing" } else { fixstr <- namepars[idparsfix] }
      if (verbose) { cat("You are fixing",fixstr,"\n") }
      if(exteq == 1) { fixstr <- "exactly" } else { fixstr <- "not" }
      if (verbose) { cat("Extinction rate of incipient species is",fixstr,"the same as for good species.\n") }
      trparsopt <- initparsopt/(1 + initparsopt)
      trparsfix <- parsfix/(1 + parsfix)
      trparsfix[parsfix == Inf] <- 1
      pars2 <- c(cond, btorph, soc, 0, methode, n_low, n_up)
      utils::flush.console()
      initloglik <- pbd_loglik_choosepar(trparsopt = trparsopt, trparsfix = trparsfix, idparsopt = idparsopt, idparsfix = idparsfix, exteq = exteq, parsfunc = parsfunc, pars2 = pars2, brts = brts, missnumspec = missnumspec)
      if (verbose) { cat("The likelihood for the initial parameter values is",initloglik,"\n") }
      utils::flush.console()
      if(initloglik == -Inf)
      {
        cat("The initial parameter values have a likelihood that is equal to 0 or below machine precision. Try again with different initial values.\n")
        out2 <- data.frame(b = -1, mu_1 = -1, lambda_1 = -1, mu_2 = -1, loglik = -1, df = -1, conv = -1)
      } else {
        if (verbose) { cat("Optimizing the likelihood - this may take a while.","\n") }
        utils::flush.console()
        optimpars <- c(tol, maxiter)
        out <- DDD::optimizer(optimmethod = optimmethod, optimpars = optimpars, fun = pbd_loglik_choosepar, trparsopt = trparsopt, trparsfix = trparsfix, idparsopt = idparsopt, idparsfix = idparsfix, exteq = exteq, parsfunc = parsfunc, pars2 = pars2, brts = brts, missnumspec = missnumspec, num_cycles = num_cycles)
        if(out$conv > 0)
        {
          cat("Optimization has not converged. Try again with different initial values.\n")
          out2 <- data.frame(b = -1, mu_1 = -1, lambda_1 = -1, mu_2 = -1, loglik = -1, df = -1, conv = -1)
        } else {
          MLtrpars <- as.numeric(unlist(out$par))
          MLpars <- MLtrpars/(1-MLtrpars)
          MLpars1 <- rep(0, 4)
          MLpars1[idparsopt] <- MLpars
          if(length(idparsfix) != 0) { MLpars1[idparsfix] <- parsfix }
          if(exteq == 1) { MLpars1[4] <- MLpars[2] }
          if(MLpars1[3] > 10^7){MLpars1[3] <- Inf}
          ML <- as.numeric(unlist(out$fvalues))
          out2 <- data.frame(b = MLpars1[1], mu_1 = MLpars1[2], lambda_1 = MLpars1[3], mu_2 = MLpars1[4], loglik = ML, df = length(initparsopt), conv = unlist(out$conv))
          if(verbose)
          {
            s1 <- sprintf('Maximum likelihood parameter estimates: b: %f, mu_1: %f, lambda_1: %f, mu_2: %f', MLpars1[1], MLpars1[2], MLpars1[3], MLpars1[4])
            s2 <- sprintf('Maximum loglikelihood: %f', ML)
            s3 <- sprintf('The expected duration of speciation for these parameters is: %f', pbd_durspec_mean(c(MLpars1[1], MLpars1[3], MLpars1[4])))
            s4 <- sprintf('The median duration of speciation for these parameters is: %f', pbd_durspec_quantile(c(MLpars1[1], MLpars1[3], MLpars1[4]), 0.5))
            cat("\n",s1,"\n",s2,"\n",s3,"\n",s4,"\n")
            utils::flush.console()
          }
        }
      }
    }
  }
  return(invisible(out2))
}


#' Mean duration of speciation under protracted birth-death model of
#' diversification
#'
#' pbd_durspec_mean computes the mean duration of speciation under the
#' protracted speciation model for a given set of parameters
#'
#'
#' @param pars Vector of parameters: \cr \cr \code{pars[1]} corresponds to b (=
#' la_3 in Etienne & Rosindell R2012) = speciation initiation rate \cr
#' \code{pars[2]} corresponds to la_1 (= la_2 in Etienne & Rosindell 2012) =
#' speciation completion rate \cr \code{pars[3]} corresponds to mu_2 (= mu_i in
#' ER2012) = extinction rate of incipient species \cr
#' @return The expected duration of speciation
#' @author Rampal S. Etienne
#' @seealso \code{\link{pbd_durspec_density}}\cr
#' \code{\link{pbd_durspec_cumdensity}}\cr \code{\link{pbd_durspec_mode}}\cr
#' \code{\link{pbd_durspec_quantile}}\cr \code{\link{pbd_durspec_moment}}\cr
#' \code{\link{pbd_durspec_var}}
#' @keywords models
#' @examples
#'  pbd_durspec_mean(pars = c(0.5,0.3,0.1))
#' @export pbd_durspec_mean
pbd_durspec_mean <- function(pars)
{
  # Do not check 'pars' being valid, as this is the classic behavior
  pbd_durspec_mean_impl(la2 = pars[2], la3 = pars[1], mu2 = pars[3])
}



#' Calculate the mean durations of speciation (equations 19 and 20 of reference
#' article)
#'
#' Calculate the mean durations of speciation (equations 19 and 20 of reference
#' article)
#'
#'
#' @param eris one or more extinction rates of the incipient species, or mu_2
#' in article, in probability per time unit. These values will be recycled if
#' needed
#' @param scrs one or more speciation completion rates, or lambda_2 in article,
#' in probability per time unit. These values will be recycled if needed
#' @param siris one or more speciation initiation rates of incipient species,
#' or lambda_3 in article, in probability per time unit. These values will be
#' recycled if needed
#' @return the means durations of speciation, in time units. Puts an NA at each
#' invalid combination of inputs
#' @author Richel J.C. Bilderbeek
#' @seealso pbd_mean_durspec
#' @references Etienne, Rampal S., and James Rosindell. "Prolonging the past
#' counteracts the pull of the present: protracted speciation can explain
#' observed slowdowns in diversification." Systematic Biology 61.2 (2012):
#' 204-213.
#' @examples
#'
#'   eris <- c(0.1, 0.2) # extinction rates of incipient species
#'   scrs <- c(0.2, 0.3)  # speciation completion rates
#'   siris <- c(0.3, 0.4) # speciation initiation rates of incipient species
#'   mean_durspecs <- pbd_mean_durspecs(eris, scrs, siris)
#'   expected_mean_durspecs <- c(2.829762, 1.865386)
#'   testthat::expect_equal(mean_durspecs, expected_mean_durspecs,
#'     tolerance = 0.000001)
#'
#' @export pbd_mean_durspecs
pbd_mean_durspecs <- function(eris, scrs, siris) {

  # Find invalid indices
  invalid <- eris < 0.0 | is.na(eris) |
    scrs < 0.0 | is.na(scrs) |
    siris < 0.0 | is.na(siris)

  # Correct invalid rates to valid ones
  correct <- function(x) {
    x[ is.na(x) | x < 0.0] <- 0.0
    x
  }

  # Get durations for corrected rates
  v <- mapply(pbd_mean_durspec, correct(eris), correct(scrs), correct(siris))

  # Let invalid rates become NAs
  v[ invalid ] <- NA
  v
}



#' Calculate the mean duration of speciation (equations 19 and 20 of reference
#' article), non-vectorized
#'
#' Calculate the mean duration of speciation (equations 19 and 20 of reference
#' article), non-vectorized
#'
#'
#' @param eri one single extinction rate of the incipient species, or mu_2 in
#' article, in probability per time unit
#' @param scr one single speciation completion rate, or lambda_2 in article, in
#' probability per time unit
#' @param siri one single speciation initiation rate of incipient species, or
#' lambda_3 in article, in probability per time unit
#' @return the means duration of speciation, in time units
#' @author Richel J.C. Bilderbeek
#' @references Etienne, Rampal S., and James Rosindell. "Prolonging the past
#' counteracts the pull of the present: protracted speciation can explain
#' observed slowdowns in diversification." Systematic Biology 61.2 (2012):
#' 204-213.
#' @examples
#'
#'   eri <- 0.1 # extinction rate of incipient species
#'   scr <- 0.2 # speciation completion rate
#'   siri <- 0.3 # speciation initiation rate of incipient species
#'   mean_durspec <- pbd_mean_durspec(eri, scr, siri)
#'   expected_mean_durspec <- 2.829762
#'   testthat::expect_equal(mean_durspec, expected_mean_durspec,
#'     tolerance = 0.000001)
#'
#' @export pbd_mean_durspec
pbd_mean_durspec <- function(eri, scr, siri) {
  if (is.na(eri) || eri < 0.0) {
    stop("extinction rate of incipient species must be zero or positive")
  }
  if (is.na(scr) || scr < 0.0) {
    stop("speciation completion rate must be zero or positive")
  }
  if (is.na(siri) || siri < 0.0) {
    stop("speciation initiation rate of incipient species must ",
         "be zero or positive")
  }
  pbd_durspec_mean_impl(la2 = scr, la3 = siri, mu2 = eri)
}




#' Actual calculation of the mean duration of speciation (equations 19 and 20
#' of reference article) assuming all inputs are correct
#'
#' Actual calculation of the mean duration of speciation (equations 19 and 20
#' of reference article) assuming all inputs are correct
#'
#'
#' @param la2 lambda_2, the speciation completion rate, in probability per time
#' unit
#' @param la3 lambda_3, speciation initiation rate of incipient species, in
#' probability per time unit
#' @param mu2 mu_2 extinction rate of the incipient species, in probability per
#' time unit
#' @author Rampal S. Etienne
#' @seealso pbd_mean_durspec
#' @references Etienne, Rampal S., and James Rosindell. "Prolonging the past
#' counteracts the pull of the present: protracted speciation can explain
#' observed slowdowns in diversification." Systematic Biology 61.2 (2012):
#' 204-213.
pbd_durspec_mean_impl <- function(la2, la3, mu2)
{
  if(la2 == Inf) {
    rho_mean <- 0.0
  } else if(la2 == 0) {
    rho_mean <- Inf
  } else if(la3 == 0) {
    rho_mean <- 1.0 / (la2 + mu2)
  } else if(mu2 == 0) {
    rho_mean <- 1/la3 * log(1 + la3/la2)
  } else {
    D <- sqrt((la2 + la3)^2 + 2*(la2 - la3) * mu2 + mu2^2)
    rho_mean <- 2/(D - la2 + la3 - mu2) * log(2 / (1 + (la2 - la3 + mu2)/D))
  }
  rho_mean
}


#' Quantiles of duration of speciation under protracted birth-death model of
#' diversification
#'
#' pbd_durspec_quantile computes a quantile of the duration of speciation under
#' the protracted speciation model for a given set of parameters
#'
#'
#' @param pars Vector of parameters: \cr \cr \code{pars[1]} corresponds to b (=
#' la_3 in Etienne & Rosindell R2012) = speciation initiation rate \cr
#' \code{pars[2]} corresponds to la_1 (= la_2 in Etienne & Rosindell 2012) =
#' speciation completion rate \cr \code{pars[3]} corresponds to mu_2 (= mu_i in
#' ER2012) = extinction rate of incipient species \cr
#' @param p Quantile (e.g. p = 0.5 gives the median)
#' @return The quantil of the duration of speciation
#' @author Rampal S. Etienne
#' @seealso \code{\link{pbd_durspec_density}}\cr
#' \code{\link{pbd_durspec_cumdensity}}\cr \code{\link{pbd_durspec_mean}}\cr
#' \code{\link{pbd_durspec_mode}}\cr \code{\link{pbd_durspec_moment}}\cr
#' \code{\link{pbd_durspec_var}}
#' @keywords models
#' @examples
#'  pbd_durspec_quantile(pars = c(0.5,0.3,0.1),0.5)
#' @export pbd_durspec_quantile
pbd_durspec_quantile <- function(pars, p)
{
  expdurspec <- pbd_durspec_mean(pars)
  if(expdurspec < 1E-7)
  {
    q <- 0
  } else {
    found <- 0
    uptau <- 100 * expdurspec
    while(found == 0)
    {
      if(pbd_durspec_cumdensity(pars,uptau) > p)
      {
        found <- 1
      } else {
        uptau <- 10*uptau
      }
    }
    q <- stats::uniroot(function(x) pbd_durspec_cumdensity(pars, x) - p, c(0, uptau))$root
  }
  return(q)
}


#' Cumulative density of duration of speciation under protracted birth-death
#' model of diversification
#'
#' pbd_durspec_cumdensity computes the cumulative density of the duration of
#' speciation under the protracted speciation model for a given set of
#' parameters
#'
#'
#' @param pars Vector of parameters: \cr \cr \code{pars[1]} corresponds to b (=
#' la_3 in Etienne & Rosindell R2012) = speciation initiation rate \cr
#' \code{pars[2]} corresponds to la_1 (= la_2 in Etienne & Rosindell 2012) =
#' speciation completion rate \cr \code{pars[3]} corresponds to mu_2 (= mu_i in
#' ER2012) = extinction rate of incipient species \cr
#' @param tau Value of the duration of speciation at which the cumulative
#' density must be computed
#' @return The cumulative density of the duration of speciation
#' @author Rampal S. Etienne
#' @seealso \code{\link{pbd_durspec_density}}\cr
#' \code{\link{pbd_durspec_mean}}\cr \code{\link{pbd_durspec_mode}}\cr
#' \code{\link{pbd_durspec_quantile}}\cr \code{\link{pbd_durspec_moment}}\cr
#' \code{\link{pbd_durspec_var}}
#' @keywords models
#' @examples
#'  pbd_durspec_cumdensity(pars = c(0.5,0.3,0.1),3)
#' @export pbd_durspec_cumdensity
pbd_durspec_cumdensity <- function(pars, tau)
{
  stats::integrate(function(x) pbd_durspec_density(pars,x),lower = 0,upper = tau, abs.tol = 1e-10)$value
}


#' Probability density for duration of speciation under protracted birth-death
#' model of diversification
#'
#' pbd_durspec_density computes the probability density of the duration of
#' speciation under the protracted speciation model for a given set of
#' parameters
#'
#'
#' @param pars Vector of parameters: \cr \cr \code{pars[1]} corresponds to b (=
#' la_3 in Etienne & Rosindell R2012) = speciation initiation rate \cr
#' \code{pars[2]} corresponds to la_1 (= la_2 in Etienne & Rosindell 2012) =
#' speciation completion rate \cr \code{pars[3]} corresponds to mu_2 (= mu_i in
#' ER2012) = extinction rate of incipient species \cr
#' @param tau The duration of speciation for which the density must be computed
#' @return The probability density
#' @author Rampal S. Etienne
#' @seealso \code{\link{pbd_durspec_cumdensity}}\cr
#' \code{\link{pbd_durspec_mean}}\cr \code{\link{pbd_durspec_mode}}\cr
#' \code{\link{pbd_durspec_quantile}}\cr \code{\link{pbd_durspec_moment}}\cr
#' \code{\link{pbd_durspec_var}}
#' @keywords models
#' @examples
#'  pbd_durspec_density(pars = c(0.5,0.3,0.1), tau = 1)
#' @export pbd_durspec_density
pbd_durspec_density <- function(pars, tau)
{
  la3 <- pars[1]
  la2 <- pars[2]
  mu2 <- pars[3]
  if(la2 < Inf)
  {
    phi <- la2 - la3 + mu2
    D <- sqrt((la2 + la3)^2 + 2*(la2 - la3) * mu2 + mu2^2)
    rho <- 2*D^2 * exp(-D*tau) * (D + phi) / (D + phi + exp(-D*tau) * (D - phi))^2
  } else {
    rho <- Inf * (tau == 0)
    rho[is.nan(rho)] <- 0
  }
  return(rho)
}


pbd_loglik_choosepar <- function(trparsopt, trparsfix, idparsopt, idparsfix, exteq, parsfunc, pars2, brts, missnumspec)
{
  trpars1 <- rep(0, 4)
  trpars1[idparsopt] <- trparsopt
  if(length(idparsfix) != 0)
  {
    trpars1[idparsfix] <- trparsfix
  }
  if(exteq == 1)
  {
    trpars1[4] <- trpars1[2]
  }
  if(max(trpars1) > 1 || min(trpars1) < 0 || max(trpars1[c(1,2,4)]) == 1)
  {
    loglik <- -Inf
  } else {
    pars1 <- trpars1/(1 - trpars1)
    loglik <- pbd_loglik(pars1, parsfunc, pars2, brts, missnumspec)
    if(is.nan(loglik) || is.na(loglik))
    {
      cat("Parameter values have been used that cause numerical problems.\n")
      loglik <- -Inf
    }
  }
  return(loglik)
}


logcondfun <- function(nn, socsoc, yy)
{
  if(socsoc == 1)
  {
    nfac <- 0
  } else
  {
    nfac <- log(nn - 1)
  }
  pnc <- nfac + (nn - socsoc) * log(1 - yy)
  return(pnc)
}

#' Loglikelihood for protracted birth-death model of diversification
#'
#' pbd_loglik computes the loglikelihood of the parameters of the protracted
#' speciation model given a set of branching times and number of missing
#' species
#'
#'
#' @param pars1 Vector of parameters: \cr \cr \code{pars1[1]} corresponds to b
#' (= la_1 in Etienne & Rosindell R2012) = speciation initiation rate \cr
#' \code{pars1[2]} corresponds to mu_1 (= mu_g in Etienne & Rosindell 2012) =
#' extinction rate of good species \cr \code{pars1[3]} corresponds to la_1 (=
#' la_2 in Etienne & Rosindell 2012) = speciation completion rate \cr
#' \code{pars1[4]} corresponds to mu_2 (= mu_i in ER2012) = extinction rate of
#' incipient species \cr When rates depend on time this time dependence should
#' be specified in pars1f and pars1 then becomes the parameters used in pars1f
#' \cr \cr
#' @param pars1f Vector of functions how the rates depend on time, default
#' functions are constant functions of the parameters in pars1: \cr \cr
#' \code{pars1f[1]} corresponds to time-dependence of b (= la_1 in Etienne &
#' Rosindell R2012) = speciation initiation rate \cr \code{pars1f[2]}
#' corresponds to time-dependence of mu_1 (= mu_g in Etienne & Rosindell 2012)
#' = extinction rate of good species \cr \code{pars1f[3]} corresponds to
#' time-dependence of la_1 (= la_2 in Etienne & Rosindell 2012) = speciation
#' completion rate \cr \code{pars1f[4]} corresponds to time-dependence of mu_2
#' (= mu_i in ER2012) = extinction rate of incipient species \cr \cr
#' @param pars2 Vector of model settings: \cr \cr \code{pars2[1]} set the
#' conditioning on non-extinction of the clade (1) or not (0) \cr \cr
#' \code{pars2[2]} sets whether the likelihood is for the branching times (0)
#' or the phylogeny (1) \cr \cr \code{pars2[3]} sets whether the first element
#' of the branching times is the stem (1) or the crown (2) age \cr \cr
#' \code{pars2[4]} sets whether the parameters and likelihood should be shown
#' on screen (1) or not (0) \cr \cr \code{pars2[5]} sets which method should be
#' used in the ode-solver. Default is 'lsoda'. See package deSolve for details.
#' \cr \cr \code{pars2[6]}Sets the lower bound of the number of species on
#' which conditioning should be done when cond = 2. Set this to 0 when
#' conditioning should be done on precisely the number of species (default)\cr
#' \cr \code{pars2[7]}Sets the upper bound of the number of species on which
#' conditioning should be done when cond = 2. Set this to 0 when conditioning
#' should be done on precisely the number of species (default)\cr \cr
#' @param brts A set of branching times of a phylogeny, all positive
#' @param missnumspec The number of species that are in the clade but missing
#' in the phylogeny
#' @return The loglikelihood
#' @author Rampal S. Etienne
#' @seealso \code{\link{pbd_ML}}
#' @keywords models
#' @examples
#'  pbd_loglik(pars1 = c(0.2,0.1,1,0.1), pars2 = c(1,1,2,0,"lsoda"),brts = 1:10)
#' @export pbd_loglik
pbd_loglik <- function(
  pars1,pars1f = c(function(pars) {pars[1]},function(pars) {pars[2]},function(pars) {pars[3]},function(pars) {pars[4]}),
  pars2 = c(1,1,2,1,"lsoda",0,0),
  brts,
  missnumspec = 0
)
{
  # pbd_loglik computes the loglikelihood of the protracted speciation model given a set of branching times and data

  # pars1 contains model parameters
  # In the simplest case where rates do not depend on time, we have
  # - pars1[1] = b (= la_1 in ER2012) = speciation initiation rate
  # - pars1[2] = mu_1 (= mu_g in ER2012) = extinction rate of good species
  # - pars1[3] = la_1 (= la_2 in ER2012) = speciation completion rate
  # - pars1[4] = mu_2 (= mu_i in ER2012) = extinction rate of incipient species
  # When rates depend on time this time dependence should be specified in pars1f and pars1 then become the parameters used in pars1f
  # pars1f contains the functions how the rates depend on time, default functions are constant functions of the parameters in pars1
  # pars2 contains settings
  # - pars2[1] = cond = conditioning on age (0), age and non-extinction of the clade (1) or age and number of extant taxa (2)
  # - pars2[2] = btorph = likelihood for branching times (0) or phylogeny (1)
  # - pars2[3] = soc = stem (1) or crown (2) age
  # - pars2[4] = printing of parameters and likelihood (1) or not (0)
  # - pars2[5] = method of the numerical integration; see package deSolve for details
  # brts = set of branching times
  # missnumspec = the number of species that belong to the same clade but are not in the phylogeny

  # Example: pbd_loglik(pars1 = c(0.1,0.05,1,0.05), brts = 1:10, missnumspec = 4)

  pars1 <- c(pars1f, pars1)

  brts <- sort(abs(brts))
  abstol <- 1e-16
  reltol <- 1e-10
  b <- pars1[[1]](brts, as.numeric(pars1[5:length(pars1)]))
  methode <- pars2[5]
  cond <- as.numeric(pars2[1])
  btorph <- as.numeric(pars2[2])
  soc <- as.numeric(pars2[3])
  S <- length(brts) + (soc - 1)
  m <- missnumspec

  probs <- c(1, 1, 0, 0)
  y <- deSolve::ode(probs, c(0, brts), pbd_loglik_rhs, c(pars1), rtol = reltol, atol = abstol, method = methode)
  if(dim(y)[1] < length(brts) + 1) { return(-Inf) }

  loglik <- (btorph == 0) * lgamma(S) +
    (cond == 0) * soc * (log(y[length(brts) + 1,2]) + log(1 - y[length(brts) + 1,3])) +
    (cond == 1) * soc * log(y[length(brts) + 1,2])
  if(length(brts) > 1)
  {
    loglik <- loglik + sum(log(b) + log(y[2:length(brts), 2]) + log(1 - y[2:length(brts), 3]))
  }
  if(cond == 2)
  {
    n_l <- as.numeric(pars2[6])
    n_u <- as.numeric(pars2[7])
    if(n_l == 0 & n_u == 0)
    {
      n_l <- S + m
      n_u <- S + m
    } else if(n_l > (S + m) | n_u < (S + m) | n_u < n_l)
    {
      cat('Lower or upper boundary not possible.\n')
      return(-Inf)
    }
    if(n_l == soc & n_u == Inf)
    {
      logcond <- -soc * log(y[length(brts) + 1, 2])
    } else if(n_u == Inf)
    {
      n_u <- n_l - 1
      n_l <- soc
      logcond <- log(y[length(brts) + 1, 2]^(-2) - sum(exp(logcondfun(n_l:n_u, soc, y[(length(brts) + 1), 2]))))
      if(logcond == -Inf)
      {
        cat('Catastrophic cancellation encountered. Trying now with 10000 as upper bound.\n')
        logcond <- log(sum(exp(logcondfun((n_u + 1):10000, soc, y[(length(brts) + 1), 2]))))
      }
    } else
    {
      logcond <- log(sum(exp(logcondfun(n_l:n_u, soc, y[(length(brts) + 1), 2]))))
    }
    loglik <- loglik - logcond
  }
  if(m > 0)
  {
    if(soc == 1)
    {
      y2 <- as.numeric(c(1 - y[2:(length(brts) + 1), 2]))
    }
    if(soc == 2)
    {
      y2 <- as.numeric(c(1 - y[2:(length(brts) + 1), 2], 1 - y[length(brts) + 1, 2]))
    }
    x <- rep(0, m + 1)
    x[1] <- 1
    for(j in 1:S)
    {
      #x = convolve(x,rev((1:(m + 1)) * (y2[j]^(0:m))),type = 'open')[1:(m + 1)]
      x <- DDD::conv(x, (1:(m + 1)) * (y2[j]^(0:m)))[1:(m+1)]
    }
    loglik <- loglik + lgamma(S + 1) + lgamma(m + 1) - lgamma(S + m + 1) + log(x[m + 1])
  }

  if(as.numeric(pars2[4]) == 1)
  {
    pastetxt <- paste('Parameters:', pars1[[5]][1], sep = ' ')
    for(i in 6:length(pars1))
    {
      pastetxt <- paste(pastetxt, pars1[[i]][1], sep = ', ')
    }
    s2 <- sprintf(', Loglikelihood: %f', loglik)
    cat(pastetxt,s2,"\n",sep = "")
    utils::flush.console()
  }

  return(as.numeric(loglik))
}