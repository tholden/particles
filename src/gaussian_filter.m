function [LIK,lik] = gaussian_filter(ReducedForm, Y, start, ParticleOptions, ThreadsOptions)
% Evaluates the likelihood of a non-linear model approximating the
% predictive (prior) and filtered (posterior) densities for state variables
% by gaussian distributions.
% Gaussian approximation is done by:
% - a spherical-radial cubature (ref: Arasaratnam & Haykin, 2009).
% - a scaled unscented transform cubature (ref: Julier & Uhlmann 1995)
% - Monte-Carlo draws from a multivariate gaussian distribution.
% First and second moments of prior and posterior state densities are computed
% from the resulting nodes/particles and allows to generate new distributions at the
% following observation.
% Pros: The use of nodes is much faster than Monte-Carlo Gaussian particle and standard particles
% filters since it treats a lesser number of particles. Furthermore, in all cases, there is no need
% of resampling.
% Cons: estimations may be biaised if the model is truly non-gaussian
% since predictive and filtered densities are unimodal.
%
% INPUTS
%    Reduced_Form     [structure] Matlab's structure describing the reduced form model.
%    Y                [double]    matrix of original observed variables.
%    start            [double]    structural parameters.
%    ParticleOptions  [structure] Matlab's structure describing options concerning particle filtering.
%    ThreadsOptions   [structure] Matlab's structure.
%
% OUTPUTS
%    LIK        [double]    scalar, likelihood
%    lik        [double]    vector, density of observations in each period.
%
% REFERENCES
%
% NOTES
%   The vector "lik" is used to evaluate the jacobian of the likelihood.
% Copyright (C) 2009-2017 Dynare Team
%
% This file is part of Dynare.
%
% Dynare is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% Dynare is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with Dynare.  If not, see <http://www.gnu.org/licenses/>.

persistent init_flag mf0 mf1
persistent nodes2 weights2 weights_c2 number_of_particles
persistent sample_size number_of_state_variables number_of_observed_variables

% Set default
if isempty(start)
    start = 1;
end

% Set persistent variables.
if isempty(init_flag)
    mf0 = ReducedForm.mf0;
    mf1 = ReducedForm.mf1;
    sample_size = size(Y,2);
    number_of_state_variables = length(mf0);
    number_of_observed_variables = length(mf1);
    number_of_particles = ParticleOptions.number_of_particles;
    init_flag = 1;
end

% compute gaussian quadrature nodes and weights on states and shocks
if isempty(nodes2)
    if ParticleOptions.distribution_approximation.cubature
        [nodes2,weights2] = spherical_radial_sigma_points(number_of_state_variables);
        weights_c2 = weights2;
    elseif ParticleOptions.distribution_approximation.unscented
        [nodes2,weights2,weights_c2] = unscented_sigma_points(number_of_state_variables,ParticleOptions);
    else
        if ~ParticleOptions.distribution_approximation.montecarlo
            error('Estimation: This approximation for the proposal is not implemented or unknown!')
        end
    end
end

if ParticleOptions.distribution_approximation.montecarlo
    set_dynare_seed('default');
end

% Get covariance matrices
Q = ReducedForm.Q;
H = ReducedForm.H;
if isempty(H)
    H = 0;
    H_lower_triangular_cholesky = 0;
else
    H_lower_triangular_cholesky = reduced_rank_cholesky(H)';
end

% Get initial condition for the state vector.
StateVectorMean = ReducedForm.StateVectorMean;
StateVectorVarianceSquareRoot = reduced_rank_cholesky(ReducedForm.StateVectorVariance)';
state_variance_rank = size(StateVectorVarianceSquareRoot,2);
Q_lower_triangular_cholesky = reduced_rank_cholesky(Q)';

% Initialization of the likelihood.
const_lik = (2*pi)^(number_of_observed_variables/2) ;
lik  = NaN(sample_size,1);
LIK  = NaN;

for t=1:sample_size
    [PredictedStateMean,PredictedStateVarianceSquareRoot,StateVectorMean,StateVectorVarianceSquareRoot] = ...
        gaussian_filter_bank(ReducedForm,Y(:,t),StateVectorMean,StateVectorVarianceSquareRoot,Q_lower_triangular_cholesky,H_lower_triangular_cholesky,H,ParticleOptions,ThreadsOptions) ;
    if ParticleOptions.distribution_approximation.cubature || ParticleOptions.distribution_approximation.unscented
        StateParticles = bsxfun(@plus,StateVectorMean,StateVectorVarianceSquareRoot*nodes2') ;
        IncrementalWeights = ...
            gaussian_densities(Y(:,t),StateVectorMean,...
                               StateVectorVarianceSquareRoot,PredictedStateMean,...
                               PredictedStateVarianceSquareRoot,StateParticles,H,const_lik,...
                               weights2,weights_c2,ReducedForm,ThreadsOptions) ;
        SampleWeights = weights2.*IncrementalWeights ;
    else
        StateParticles = bsxfun(@plus,StateVectorVarianceSquareRoot*randn(state_variance_rank,number_of_particles),StateVectorMean) ;
        IncrementalWeights = ...
            gaussian_densities(Y(:,t),StateVectorMean,...
                               StateVectorVarianceSquareRoot,PredictedStateMean,...
                               PredictedStateVarianceSquareRoot,StateParticles,H,const_lik,...
                               1/number_of_particles,1/number_of_particles,ReducedForm,ThreadsOptions) ;
        SampleWeights = IncrementalWeights/number_of_particles ;
    end
    SampleWeights = SampleWeights + 1e-6*ones(size(SampleWeights,1),1) ;
    SumSampleWeights = sum(SampleWeights) ;
    lik(t) = log(SumSampleWeights) ;
    SampleWeights = SampleWeights./SumSampleWeights ;
    if not(ParticleOptions.distribution_approximation.cubature || ParticleOptions.distribution_approximation.unscented)
        if (ParticleOptions.resampling.status.generic && neff(SampleWeights)<ParticleOptions.resampling.threshold*sample_size) || ParticleOptions.resampling.status.systematic
            StateParticles = resample(StateParticles',SampleWeights,ParticleOptions)' ;
            SampleWeights = ones(number_of_particles,1)/number_of_particles;
        end
    end
    StateVectorMean = StateParticles*SampleWeights ;
    temp = bsxfun(@minus,StateParticles,StateVectorMean) ;
    StateVectorVarianceSquareRoot = reduced_rank_cholesky( bsxfun(@times,SampleWeights',temp)*temp' )';
end

LIK = -sum(lik(start:end));
