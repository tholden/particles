function [LIK,lik] = sequential_importance_particle_filter(ReducedForm,Y,start,ParticleOptions,ThreadsOptions)

% Evaluates the likelihood of a nonlinear model with a particle filter (optionally with resampling).

% Copyright (C) 2011-2015 Dynare Team
%
% This file is part of Dynare (particles module).
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Dynare particles module is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <http://www.gnu.org/licenses/>.

persistent init_flag
persistent mf0 mf1
persistent number_of_particles number_of_state_variables
persistent sample_size number_of_observed_variables number_of_structural_innovations

% Set default value for start
if isempty(start)
    start = 1;
end

% Set flag for prunning
pruning = ParticleOptions.pruning;

% Get steady state and mean.
steadystate = ReducedForm.steadystate;
constant = ReducedForm.constant;
state_variables_steady_state = ReducedForm.state_variables_steady_state;

% Set persistent variables (if needed).
if isempty(init_flag)
    mf0 = ReducedForm.mf0;
    mf1 = ReducedForm.mf1;
    sample_size = size(Y,2);
    number_of_state_variables = length(mf0);
    number_of_observed_variables = length(mf1);
    number_of_structural_innovations = length(ReducedForm.Q);
    number_of_particles = ParticleOptions.number_of_particles;
    init_flag = 1;
end

% Set local state space model (first order approximation).
ghx  = ReducedForm.ghx;
ghu  = ReducedForm.ghu;

% Set local state space model (second order approximation).
ghxx = ReducedForm.ghxx;
ghuu = ReducedForm.ghuu;
ghxu = ReducedForm.ghxu;

% Get covariance matrices.
Q = ReducedForm.Q; % Covariance matrix of the structural innovations.
H = ReducedForm.H; % Covariance matrix of the measurement errors.
if isempty(H)
    H = 0;
end

% Initialization of the likelihood.
const_lik = log(2*pi)*number_of_observed_variables +log(det(H)) ;
lik  = NaN(sample_size,1);

% Get initial condition for the state vector.
StateVectorMean = ReducedForm.StateVectorMean;
StateVectorVarianceSquareRoot = chol(ReducedForm.StateVectorVariance)';%reduced_rank_cholesky(ReducedForm.StateVectorVariance)';
if pruning
    StateVectorMean_ = StateVectorMean;
    StateVectorVarianceSquareRoot_ = StateVectorVarianceSquareRoot;
end

% Get the rank of StateVectorVarianceSquareRoot
state_variance_rank = size(StateVectorVarianceSquareRoot,2);

% Factorize the covariance matrix of the structural innovations
Q_lower_triangular_cholesky = chol(Q)';

% Set seed for randn().
set_dynare_seed('default');

% Initialization of the weights across particles.
weights = ones(1,number_of_particles)/number_of_particles ;
StateVectors = bsxfun(@plus,StateVectorVarianceSquareRoot*randn(state_variance_rank,number_of_particles),StateVectorMean);
if pruning
    StateVectors_ = StateVectors;
end

% Loop over observations
for t=1:sample_size
    yhat = bsxfun(@minus,StateVectors,state_variables_steady_state);
    epsilon = Q_lower_triangular_cholesky*randn(number_of_structural_innovations,number_of_particles);
    if pruning
        yhat_ = bsxfun(@minus,StateVectors_,state_variables_steady_state);
        [tmp, tmp_] = local_state_space_iteration_2(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu,yhat_,steadystate,ThreadsOptions.local_state_space_iteration_2);
    else
        tmp = local_state_space_iteration_2(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu,ThreadsOptions.local_state_space_iteration_2);
    end
    %PredictedObservedMean = tmp(mf1,:)*transpose(weights);
    PredictionError = bsxfun(@minus,Y(:,t),tmp(mf1,:));
    %dPredictedObservedMean = bsxfun(@minus,tmp(mf1,:),PredictedObservedMean);
    %PredictedObservedVariance = bsxfun(@times,dPredictedObservedMean,weights)*dPredictedObservedMean' + H;
    %PredictedObservedVariance = H;
    if rcond(H) > 1e-16
        lnw = -.5*(const_lik+sum(PredictionError.*(H\PredictionError),1));
    else
        LIK = NaN;
        return
    end
    dfac = max(lnw);
    wtilde = weights.*exp(lnw-dfac);
    lik(t) = log(sum(wtilde))+dfac;
    weights = wtilde/sum(wtilde);
    if (ParticleOptions.resampling.status.generic && neff(weights)<ParticleOptions.resampling.threshold*sample_size) || ParticleOptions.resampling.status.systematic
        if pruning
            temp = resample([tmp(mf0,:)' tmp_(mf0,:)'],weights',ParticleOptions);
            StateVectors = temp(:,1:number_of_state_variables)';
            StateVectors_ = temp(:,number_of_state_variables+1:2*number_of_state_variables)';
        else
            StateVectors = resample(tmp(mf0,:)',weights',ParticleOptions)';
        end
        weights = ones(1,number_of_particles)/number_of_particles;
    elseif ParticleOptions.resampling.status.none
        StateVectors = tmp(mf0,:);
        if pruning
            StateVectors_ = tmp_(mf0,:);
        end
    end
end

LIK = -sum(lik(start:end));