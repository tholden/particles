function [LIK,lik] = gaussian_mixture_filter(ReducedForm,Y,start,DynareOptions)
% Evaluates the likelihood of a non-linear model approximating the state
% variables distributions with gaussian mixtures. Gaussian Mixture allows reproducing
% a wide variety of generalized distributions (when multimodal for instance).
% Each gaussian distribution is obtained whether
%   - with a Smolyak quadrature à la Kronrod & Paterson (Heiss & Winschel 2010, Winschel & Kratzig 2010).
%   - with a radial-spherical cubature
%   - with scaled unscented sigma-points
% A Sparse grid Kalman Filter is implemented on each component of the mixture,
% which confers it a weight about current information.
% Information on the current observables is then embodied in the proposal
% distribution in which we draw particles, which allows
%   - reaching a greater precision relatively to a standard particle filter,
%   - reducing the number of particles needed,
%   - still being faster.
%
%
% INPUTS
%    reduced_form_model     [structure] Matlab's structure describing the reduced form model.
%                                       reduced_form_model.measurement.H   [double]   (pp x pp) variance matrix of measurement errors.
%                                       reduced_form_model.state.Q         [double]   (qq x qq) variance matrix of state errors.
%                                       reduced_form_model.state.dr        [structure] output of resol.m.
%    Y                      [double]    pp*smpl matrix of (detrended) data, where pp is the maximum number of observed variables.
%    start                  [integer]   scalar, likelihood evaluation starts at 'start'.
%
% OUTPUTS
%    LIK        [double]    scalar, likelihood
%    lik        [double]    vector, density of observations in each period.
%
% REFERENCES
%
% Van der Meerwe & Wan, Gaussian Mixture Sigma-Point Particle Filters for Sequential
% Probabilistic Inference in Dynamic State-Space Models.
% Heiss & Winschel, 2010, Journal of Applied Economics.
% Winschel & Kratzig, 2010, Econometrica.
%
% NOTES
%   The vector "lik" is used to evaluate the jacobian of the likelihood.
% Copyright (C) 2009-2013 Dynare Team
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


persistent init_flag mf0 mf1 Gprime Gsecond
persistent nodes weights weights_c I J G number_of_particles
persistent sample_size number_of_state_variables number_of_observed_variables number_of_structural_innovations

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
  number_of_structural_innovations = length(ReducedForm.Q);
  G = DynareOptions.particle.mixture_state_variables;           % number of GM components in state
  I = DynareOptions.particle.mixture_structural_shocks ;        % number of GM components in structural noise
  J = DynareOptions.particle.mixture_measurement_shocks ;       % number of GM components in observation noise
  Gprime = G*I ;
  Gsecond = G*I*J ;
  number_of_particles = DynareOptions.particle.number_of_particles;
  init_flag = 1;
end

SampleWeights = ones(Gsecond,1)/Gsecond ;

% compute gaussian quadrature nodes and weights on states and shocks
if isempty(nodes)
    if DynareOptions.particle.distribution_approximation.cubature
        [nodes,weights] = spherical_radial_sigma_points(number_of_state_variables);
        weights_c = weights;
    elseif DynareOptions.particle.distribution_approximation.unscented
        [nodes,weights,weights_c] = unscented_sigma_points(number_of_state_variables,DynareOptions);
    else
        if ~DynareOptions.particle.distribution_approximation.montecarlo
            error('Estimation: This approximation for the proposal is not implemented or unknown!')
        end
    end
end

if DynareOptions.particle.distribution_approximation.montecarlo
    set_dynare_seed('default');
    SampleWeights = 1/number_of_particles ;
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
Q_lower_triangular_cholesky = reduced_rank_cholesky(Q)';

% Initialize all matrices
StateWeights = ones(1,G)/G ;
StateMu = ReducedForm.StateVectorMean*ones(1,G) ;
StateSqrtP = zeros(number_of_state_variables,number_of_state_variables,G) ;
for g=1:G
  StateSqrtP(:,:,g) = reduced_rank_cholesky(ReducedForm.StateVectorVariance)' ;
end

StructuralShocksWeights = ones(1,I)/I ;
StructuralShocksMu = zeros(number_of_structural_innovations,I) ;
StructuralShocksSqrtP = zeros(number_of_structural_innovations,number_of_structural_innovations,I) ;
for i=1:I
  StructuralShocksSqrtP(:,:,i) = Q_lower_triangular_cholesky ;
end

ObservationShocksWeights  = ones(1,J)/J ;
ObservationShocksMu = zeros(number_of_observed_variables,J) ;
ObservationShocksSqrtP = zeros(number_of_observed_variables,number_of_observed_variables,J) ;
for j=1:J
  ObservationShocksSqrtP(:,:,j) = H_lower_triangular_cholesky ;
end

StateWeightsPrior = zeros(1,Gprime) ;
StateMuPrior = zeros(number_of_state_variables,Gprime) ;
StateSqrtPPrior = zeros(number_of_state_variables,number_of_state_variables,Gprime) ;

StateWeightsPost = zeros(1,Gsecond) ;
StateMuPost = zeros(number_of_state_variables,Gsecond) ;
StateSqrtPPost = zeros(number_of_state_variables,number_of_state_variables,Gsecond) ;

%estimate = zeros(sample_size,number_of_state_variables,3) ;
const_lik = (2*pi)^(.5*number_of_observed_variables) ;

ks = 0 ;
lik  = NaN(sample_size,1);
LIK  = NaN;
for t=1:sample_size
    % Build the proposal joint quadratures of Gaussian on states, structural
    % shocks and observation shocks based on each combination of mixtures
    for i=1:I
        for j=1:J
            for g=1:G ;
                a = g + (j-1)*G ;
                b = a + (i-1)*Gprime ;
                [StateMuPrior(:,a),StateSqrtPPrior(:,:,a),StateWeightsPrior(1,a),...
                 StateMuPost(:,b),StateSqrtPPost(:,:,b),StateWeightsPost(1,b)] =...
                 gaussian_mixture_filter_bank(ReducedForm,Y(:,t),StateMu(:,g),StateSqrtP(:,:,g),StateWeights(1,g),...
                                                                 StructuralShocksMu(:,i),StructuralShocksSqrtP(:,:,i),StructuralShocksWeights(1,i),...
                                                                 ObservationShocksMu(:,j),ObservationShocksSqrtP(:,:,j),ObservationShocksWeights(1,j),...
                                                                 H,H_lower_triangular_cholesky,const_lik,DynareOptions) ;
            end
        end
    end

    % Normalize weights
    StateWeightsPrior = StateWeightsPrior/sum(StateWeightsPrior,2) ;
    StateWeightsPost = StateWeightsPost/sum(StateWeightsPost,2) ;

    if DynareOptions.particle.distribution_approximation.cubature || DynareOptions.particle.distribution_approximation.unscented
        for i=1:Gsecond
            StateParticles = bsxfun(@plus,StateMuPost(:,i),StateSqrtPPost(:,:,i)*nodes') ;
            IncrementalWeights = gaussian_mixture_densities(Y(:,t),StateMuPrior,StateSqrtPPrior,StateWeightsPrior,...
                                                                   StateMuPost,StateSqrtPPost,StateWeightsPost,...
                                                                   StateParticles,H,const_lik,weights,weights_c,ReducedForm,DynareOptions) ;
            SampleWeights(i) = sum(StateWeightsPost(i)*weights.*IncrementalWeights) ;
        end
        SumSampleWeights = sum(SampleWeights) ;
        lik(t) = log(SumSampleWeights) ;
        SampleWeights = SampleWeights./SumSampleWeights ;
        [ras,SortedRandomIndx] = sort(rand(1,Gsecond));
        SortedRandomIndx = SortedRandomIndx(1:G);
        indx = index_resample(0,SampleWeights,DynareOptions) ;
        indx = indx(SortedRandomIndx) ;
        StateMu = StateMuPost(:,indx);
        StateSqrtP = StateSqrtPPost(:,:,indx);
        StateWeights = ones(1,G)/G ;
    else
        % Sample particle in the proposal distribution, ie the posterior state GM
        StateParticles = importance_sampling(StateMuPost,StateSqrtPPost,StateWeightsPost',number_of_particles) ;
        % Compute prior, proposal and likelihood of particles
        IncrementalWeights = gaussian_mixture_densities(Y(:,t),StateMuPrior,StateSqrtPPrior,StateWeightsPrior,...
                                                               StateMuPost,StateSqrtPPost,StateWeightsPost,...
                                                               StateParticles,H,const_lik,1/number_of_particles,...
                                                               1/number_of_particles,ReducedForm,DynareOptions) ;
        % calculate importance weights of particles
        SampleWeights = SampleWeights.*IncrementalWeights ;
        SumSampleWeights = sum(SampleWeights,1) ;
        SampleWeights = SampleWeights./SumSampleWeights ;
        lik(t) = log(SumSampleWeights) ;
        % First possible state point estimates
        %estimate(t,:,1) = SampleWeights*StateParticles' ;
        % Resampling if needed of required
        Neff = 1/sum(bsxfun(@power,SampleWeights,2)) ;
        if (DynareOptions.particle.resampling.status.generic && Neff<.5*sample_size) || DynareOptions.particle.resampling.status.systematic
            ks = ks + 1 ;
            StateParticles = resample(StateParticles',SampleWeights,DynareOptions)' ;
            StateVectorMean = mean(StateParticles,2) ;
            StateVectorVarianceSquareRoot = reduced_rank_cholesky( (StateParticles*StateParticles')/number_of_particles - StateVectorMean*(StateVectorMean') )';
            SampleWeights = 1/number_of_particles ;
        elseif DynareOptions.particle.resampling.status.none
            StateVectorMean = StateParticles*sampleWeights ;
            temp = sqrt(SampleWeights').*StateParticles ;
            StateVectorVarianceSquareRoot = reduced_rank_cholesky( temp*temp' - StateVectorMean*(StateVectorMean') )';
        end
        % Use the information from particles to update the gaussian mixture on state variables
        [StateMu,StateSqrtP,StateWeights] = fit_gaussian_mixture(StateParticles,StateMu,StateSqrtP,StateWeights,0.001,10,1) ;
        %estimate(t,:,3) = StateWeights*StateMu' ;
    end
end

LIK = -sum(lik(start:end)) ;