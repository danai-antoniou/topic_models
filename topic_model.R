####################################
# Compute optimal number of topics #
####################################
# Harmonic mean estimator: use the sample log likelihoods to approximate 
# p(w|T) using the harmonic mean.
# The package Rmpfr is used to deal with small numbers
# as suggested by the topicmodels author Bettina Grun.
# It is used to move numbers in a range with 
# higher precision numbers by +/  the median at certain locations .
hm_comp <-  function(ll) {
  median.ll <-  median(ll) 
  as.double(median.ll - 
              log(mean(exp(mpfr(ll , prec = 2000) + median.ll))))
}

# We parallelise the computation for efficiency 
# Leave one CPU free:
cl <-  makeCluster(detectCores()-1)
registerDoParallel(cl)

# Candidate number of topics to compute the Harmonic Mean estimator for
possible_T <-  c(seq(2,5,by=1), seq(10, 50, by=10))

# LDA parameters :
burnin <-  500 # Number of burnin iterations
iter <-  4000 # Number of iteratons for chains
seed <-  1234 # Set seed for reproducibility
keep <-  1 # Save the loglikelihood at every 'keep' iteration # The seed numbers for the other 2 Markov Chains we run
# are 3108 and 4183.
# Pass the topicmodels library to the clusters
clusterEvalQ(cl , {library(topicmodels)})

# Parallelise computation of LDA models for different number of topics T
parallel_LDA <-  foreach(i = seq_along(possible_T) , .combine=c) %dopar% { # Number of topics referred to as k in topicmodels
  T <-  possible_T[i]
  # The R package topicmodels refers to our beta parameter as delta # and to our T parameter as k.
  
  fit_LDA <-  LDA(prop_dtm, k = T, method = "Gibbs", control=list (seed = seed , burnin = burnin , iter = iter , keep=keep,alpha=50/T, delta=10))
  return(fit_LDA)}
stopCluster(cl)

# Extract the loglikelihoods per candidate topic number after burnin
all_ll <-  lapply(parallel_LDA, function(l) l@logLiks[ seq(burnin)])


# Compute the harmonic mean for each T
all_hm <-  sapply(all_ll , function(x) hm_comp(x))
dat <-  as.data.frame(cbind(possible_T=possible_T, hm=all_hm))
# The same procedure is repeated for the other two seed numbers and # and the results are gathered in table 'three.chains '.

# Plot marginal log-likelihood, log(p(w|T)), per topic number and Markov chain
ggplot(three.chains, aes(x=possible_T, y=hm)) + geom_point(aes(colour=Chain , group=Chain)) +
  stat_summary(aes(y = hm,group=1), fun.y=mean, colour="black", geom="line",group=1)+ xlab("Number of Topics T") + ylab("logP(w|T)")

# Set optimal number of topics based on the above figure
k=5

# Run LDA with the optimal number of topics k and chain 3
prop_lda <-  LDA(prop_dtm, k = k, control=list (seed = 1234, keep=keep , verbose=5, burnin = burnin , iter = iter , alpha=50/k, delta=10), method = " Gibbs " )
transf_lda <-  tidy(prop_lda)
# Plot per topic per word probabilities .
# The parameter beta is referred to as phi in our study. 
per_topic_probs <-  setDT(transf_lda)[order(topic, beta),
.SD[seq(15)], by=topic]

# Plot the top 15 terms per LDA topic
ggplot(per_topic_probs , aes(reorder(term , beta) , beta , fill = as.factor(topic))) +
  geom_col(show.legend = F) +
  coord_flip () +
  facet_wrap(~ topic , ncol = 5, scales = "free")+ labs(title = "Top 15 terms in
  each LDA topic") +
  xlab (NULL)+ylab (expression(phi))


# Log-likelihood with T=5 in all Gibbs sampling iterations
# and post burn-in period for all three Markov Chainss
ll = prop_lda@logLiks
ll.df = as.data.frame(cbind(ll=ll , iter=seq(burnin+iter))) ll.burnin = ll[(burnin+1):(burnin+iter)]
ll.df.burnin = as.data.frame(cbind(ll=ll.burnin, iter=seq(iter)))

plot1=ggplot(data=ll.df , aes(x=iter , y=ll , group=1)) + geom_line() + xlab("Iteration") + ylab("Log Likelihood")
plot2=ggplot(data=ll.df.burnin , aes(x=iter , y=ll , group=1)) + geom_line() + xlab("Iteration After Burn in") + ylab("Log Likelihood")
grid.arrange(plot1 , plot2 , ncol=2)