function [LIK,lik] = auxiliary_particle_filter(ReducedForm,Y,start,ParticleOptions,ThreadsOptions)

% Evaluates the likelihood of a nonlinear model with the auxiliary particle filter
% allowing eventually resampling.
%
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
persistent sample_size number_of_observed_variables number_of_structural_innovations

% Set default
if isempty(start)
    start = 1;
end

% Set flag for prunning
pruning = ParticleOptions.pruning;

% Get steady state and mean.
steadystate = ReducedForm.steadystate;
constant = ReducedForm.constant;
state_variables_steady_state = ReducedForm.state_variables_steady_state;

% Set persistent variables.
if isempty(init_flag)
    mf0 = ReducedForm.mf0;
    mf1 = ReducedForm.mf1;
    sample_size = size(Y,2);
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

% Get initial condition for the state vector.
StateVectorMean = ReducedForm.StateVectorMean;
StateVectorVarianceSquareRoot = chol(ReducedForm.StateVectorVariance)';
state_variance_rank = size(StateVectorVarianceSquareRoot,2);
Q_lower_triangular_cholesky = chol(Q)';

% Set seed for randn().
set_dynare_seed('default');

% Initialization of the likelihood.
const_lik = log(2*pi)*number_of_observed_variables+log(det(H));
lik  = NaN(sample_size,1);
LIK  = NaN;

% Initialization of the weights across particles.
weights = ones(1,number_of_particles)/number_of_particles ;
StateVectors = bsxfun(@plus,StateVectorVarianceSquareRoot*randn(state_variance_rank,number_of_particles),StateVectorMean);
%StateVectors = bsxfun(@plus,zeros(state_variance_rank,number_of_particles),StateVectorMean);
if pruning
    StateVectors_ = StateVectors;
end

% Uncomment for building the mean average predictions based on a sparse
% grids of structural shocks. Otherwise, all shocks are set to 0 in the
% prediction.
% if ParticleOptions.proposal_approximation.cubature
%     [nodes,nodes_weights] = spherical_radial_sigma_points(number_of_structural_innovations) ;
%     nodes_weights = ones(size(nodes,1),1)*nodes_weights ;
% elseif ParticleOptions.proposal_approximation.unscented
%     [nodes,nodes_weights,nodes_weights_c] = unscented_sigma_points(number_of_structural_innovations,ParticleOptions);
% else
%     error('Estimation: This approximation for the proposal is not implemented or unknown!')
% end
% nodes = (Q_lower_triangular_cholesky*(nodes'))' ;

nodes = zeros(1,number_of_structural_innovations) ;
nodes_weights = ones(number_of_structural_innovations,1) ;

for t=1:sample_size
    yhat = bsxfun(@minus,StateVectors,state_variables_steady_state);
    if pruning
        yhat_ = bsxfun(@minus,StateVectors_,state_variables_steady_state);
        tmp = 0 ;
        tmp_ = 0 ;
        for i=1:size(nodes)
            [tmp1, tmp1_] = local_state_space_iteration_2(yhat,nodes(i,:)'*ones(1,number_of_particles),ghx,ghu,constant,ghxx,ghuu,ghxu,yhat_,steadystate,ThreadsOptions.local_state_space_iteration_2);
            tmp = tmp + nodes_weights(i)*tmp1 ;
            tmp_ = tmp_ + nodes_weights(i)*tmp1_ ;
        end
    else
        tmp = 0 ;
        for i=1:size(nodes)
            tmp = tmp + nodes_weights(i)*local_state_space_iteration_2(yhat,nodes(i,:)'*ones(1,number_of_particles),ghx,ghu,constant,ghxx,ghuu,ghxu,ThreadsOptions.local_state_space_iteration_2);
        end
    end
    PredictionError = bsxfun(@minus,Y(:,t),tmp(mf1,:));
    %tau_tilde = weights.*(exp(-.5*(const_lik+sum(PredictionError.*(H\PredictionError),1))) + 1e-99) ;
    % Replace Gaussian density with a Student density with 3 degrees of
    % freedom for fat tails.
    z = sum(PredictionError.*(H\PredictionError),1) ;
    tau_tilde = weights.*(tpdf(z,3*ones(size(z)))+1e-99) ;
    tau_tilde = tau_tilde/sum(tau_tilde) ;
    indx = resample(0,tau_tilde',ParticleOptions);
    if pruning
        yhat_ = yhat_(:,indx) ;
    end
    yhat = yhat(:,indx) ;
    weights_stage_1 = weights(indx)./tau_tilde(indx) ;
    epsilon = Q_lower_triangular_cholesky*randn(number_of_structural_innovations,number_of_particles);
    if pruning
        [tmp, tmp_] = local_state_space_iteration_2(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu,yhat_,steadystate,ThreadsOptions.local_state_space_iteration_2);
        StateVectors_ = tmp_(mf0,:);
    else
        tmp = local_state_space_iteration_2(yhat,epsilon,ghx,ghu,constant,ghxx,ghuu,ghxu,ThreadsOptions.local_state_space_iteration_2);
    end
    StateVectors = tmp(mf0,:);
    PredictionError = bsxfun(@minus,Y(:,t),tmp(mf1,:));
    weights_stage_2 = weights_stage_1.*(exp(-.5*(const_lik+sum(PredictionError.*(H\PredictionError),1))) + 1e-99) ;
    lik(t) = log(mean(weights_stage_2)) ;
    weights = weights_stage_2/sum(weights_stage_2);
    if (ParticleOptions.resampling.status.generic && neff(weights)<ParticleOptions.resampling.threshold*sample_size) || ParticleOptions.resampling.status.systematic
        if pruning
            temp = resample([StateVectors' StateVectors_'],weights',ParticleOptions);
            StateVectors = temp(:,1:number_of_state_variables)';
            StateVectors_ = temp(:,number_of_state_variables+1:2*number_of_state_variables)';
        else
            StateVectors = resample(StateVectors',weights',ParticleOptions)';
        end
        weights = ones(1,number_of_particles)/number_of_particles;
    end
end

%plot(lik) ;
LIK = -sum(lik(start:end));
