    function [tlogpostkern,loglik] = tempered_likelihood(TargetFun,xparam1,lambda,dataset_,dataset_info,options_,M_,estim_params_,bayestopt_,bounds,oo_)
       logpostkern = -feval(TargetFun,xparam1,dataset_,dataset_info,options_,M_,estim_params_,bayestopt_,bounds,oo_);
       logprior = priordens(xparam1,bayestopt_.pshape,bayestopt_.p6,bayestopt_.p7,bayestopt_.p3,bayestopt_.p4);
       loglik = logpostkern-logprior ;
       tlogpostkern = lambda*loglik + logprior;
