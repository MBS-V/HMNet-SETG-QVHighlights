from src.evaluation.bootstrap import bootstrap_ci,bootstrap_difference_in_differences

def test_bootstrap_ci_and_did():
 ci=bootstrap_ci([1,2,3,4],n_resamples=100,seed=1)
 assert ci['lower']<=ci['estimate']<=ci['upper']
 did=bootstrap_difference_in_differences([.5,.6,.4],[.6,.7,.5],[.4,.5,.3],[.7,.8,.6],n_resamples=100,seed=2)
 assert abs(did['estimate']-.2)<1e-8 and did['n_resamples']==100
