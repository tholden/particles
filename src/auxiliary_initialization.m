function initial_distribution = auxiliary_initialization(ReducedForm,Y,start,ParticleOptions,ThreadsOptions)

% Evaluates the likelihood of a nonlinear model with a particle filter allowing eventually resampling.

% Copyright (C) 2011-2017 Dynare Team
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

persistent init_flag mf0 mf1 number_of_particles
persistent number_of_observed_variables number_of_structural_innovations

% Set default
if isempty(start)
    start = 1;
end

% Set flag for prunning
%pruning = ParticleOptions.pruning;

% Get steady state and mean.
%steadystate = ReducedForm.steadystate;
constant = ReducedForm.constant;
state_variables_steady_state = ReducedForm.state_variables_steady_state;

% Set persistent variables.
if isempty(init_flag)
    mf0 = ReducedForm.mf0;
    mf1 = ReducedForm.mf1;
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

% Get covariance matrices
Q = ReducedForm.Q;
H = ReducedForm.H;
if isempty(H)
    H = 0;
end

% Get initial condition for the state vector.
StateVectorMean = ReducedForm.StateVectorMean;
StateVectorVarianceSquareRoot = reduced_rank_cholesky(ReducedForm.StateVectorVariance)';
state_variance_rank = size(StateVectorVarianceSquareRoot,2);
%Q_lower_triangular_cholesky = chol(Q)';
%if pruning
%    StateVectorMean_ = StateVectorMean;
%    StateVectorVarianceSquareRoot_ = StateVectorVarianceSquareRoot;
%end

% Set seed for randn().
set_dynare_seed('default');

% Initialization of the likelihood.
const_lik = log(2*pi)*number_of_observed_variables;

% Initialization of the weights across particles.
weights = ones(1,number_of_particles)/number_of_particles ;
StateVectors = bsxfun(@plus,StateVectorVarianceSquareRoot*randn(state_variance_rank,number_of_particles),StateVectorMean);
%if pruning
%    StateVectors_ = StateVectors;
%end
yhat = bsxfun(@minus,StateVectors,state_variables_steady_state);
%if pruning
%    yhat_ = bsxfun(@minus,StateVectors_,state_variables_steady_state);
%    [tmp, tmp_] = local_state_space_iteration_2(yhat,zeros(number_of_structural_innovations,number_of_particles),ghx,ghu,constant,ghxx,ghuu,ghxu,yhat_,steadystate,ThreadsOptions.local_state_space_iteration_2);
%else
tmp = local_state_space_iteration_2(yhat,zeros(number_of_structural_innovations,number_of_particles),ghx,ghu,constant,ghxx,ghuu,ghxu,ThreadsOptions.local_state_space_iteration_2);
%end
PredictedObservedMean = weights*(tmp(mf1,:)');
PredictionError = bsxfun(@minus,Y(:,t),tmp(mf1,:));
dPredictedObservedMean = bsxfun(@minus,tmp(mf1,:),PredictedObservedMean');
PredictedObservedVariance = bsxfun(@times,weights,dPredictedObservedMean)*dPredictedObservedMean' + H;
wtilde = exp(-.5*(const_lik+log(det(PredictedObservedVariance))+sum(PredictionError.*(PredictedObservedVariance\PredictionError),1))) ;
tau_tilde = weights.*wtilde ;
tau_tilde = tau_tilde/sum(tau_tilde);
initial_distribution = resample(StateVectors',tau_tilde',ParticleOptions)' ;