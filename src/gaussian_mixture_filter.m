function [LIK,lik] = gaussian_mixture_filter(ReducedForm,Y,start,ParticleOptions,ThreadsOptions)
% Evaluates the likelihood of a non-linear model approximating the state
% variables distributions with gaussian mixtures. Gaussian Mixture allows reproducing
% a wide variety of generalized distributions (when multimodal for instance).
% Each gaussian distribution is obtained whether
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
    G = ParticleOptions.mixture_state_variables;           % number of GM components in state
    number_of_particles = ParticleOptions.number_of_particles;
    init_flag = 1;
end

% compute gaussian quadrature nodes and weights on states and shocks
if isempty(nodes)
    if ParticleOptions.distribution_approximation.cubature
        [nodes,weights] = spherical_radial_sigma_points(number_of_state_variables);
        weights_c = weights;
    elseif ParticleOptions.distribution_approximation.unscented
        [nodes,weights,weights_c] = unscented_sigma_points(number_of_state_variables,ParticleOptions);
    else
        if ~ParticleOptions.distribution_approximation.montecarlo
            error('Estimation: This approximation for the proposal is not implemented or unknown!')
        end
    end
end

if ParticleOptions.distribution_approximation.montecarlo
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

% Initialize mixtures
StateWeights = ones(1,G)/G ;
StateMu = ReducedForm.StateVectorMean ;
StateSqrtP = zeros(number_of_state_variables,number_of_state_variables,G) ;
temp = reduced_rank_cholesky(ReducedForm.StateVectorVariance)' ;
StateMu = bsxfun(@plus,StateMu,bsxfun(@times,diag(temp),(-(G-1)/2:1:(G-1)/2))/10) ;
for g=1:G
    StateSqrtP(:,:,g) = temp/sqrt(G) ;
end

% if ParticleOptions.mixture_structural_shocks==1
%     StructuralShocksMu = zeros(1,number_of_structural_innovations) ;
%     StructuralShocksWeights = 1 ;
% else
%     if ParticleOptions.proposal_approximation.cubature
%         [StructuralShocksMu,StructuralShocksWeights] = spherical_radial_sigma_points(number_of_structural_innovations);
%         StructuralShocksWeights = ones(size(StructuralShocksMu,1),1)*StructuralShocksWeights ;
%     elseif ParticleOptions.proposal_approximation.unscented
%         [StructuralShocksMu,StructuralShocksWeights,raf] = unscented_sigma_points(number_of_structural_innovations,ParticleOptions);
%     else
%         if ~ParticleOptions.distribution_approximation.montecarlo
%             error('Estimation: This approximation for the proposal is not implemented or unknown!')
%         end
%     end
% end
% I = size(StructuralShocksWeights,1) ;
% StructuralShocksMu = Q_lower_triangular_cholesky*(StructuralShocksMu') ;
% StructuralShocksSqrtP = zeros(number_of_structural_innovations,number_of_structural_innovations,I) ;
% for i=1:I
%   StructuralShocksSqrtP(:,:,i) = Q_lower_triangular_cholesky/sqrt(StructuralShocksWeights(i)) ;
% end
%
% if ParticleOptions.mixture_measurement_shocks==1
%     ObservationShocksMu = zeros(1,number_of_observed_variables) ;
%     ObservationShocksWeights = 1 ;
% else
%     if ParticleOptions.proposal_approximation.cubature
%         [ObservationShocksMu,ObservationShocksWeights] = spherical_radial_sigma_points(number_of_observed_variables);
%         ObservationShocksWeights = ones(size(ObservationShocksMu,1),1)*ObservationShocksWeights;
%     elseif ParticleOptions.proposal_approximation.unscented
%         [ObservationShocksMu,ObservationShocksWeights,raf] = unscented_sigma_points(number_of_observed_variables,ParticleOptions);
%     else
%         if ~ParticleOptions.distribution_approximation.montecarlo
%             error('Estimation: This approximation for the proposal is not implemented or unknown!')
%         end
%     end
% end
% J = size(ObservationShocksWeights,1) ;
% ObservationShocksMu = H_lower_triangular_cholesky*(ObservationShocksMu') ;
% ObservationShocksSqrtP = zeros(number_of_observed_variables,number_of_observed_variables,J) ;
% for j=1:J
%   ObservationShocksSqrtP(:,:,j) = H_lower_triangular_cholesky/sqrt(ObservationShocksWeights(j)) ;
% end

if ParticleOptions.mixture_structural_shocks==0
    StructuralShocksMu = zeros(1,number_of_structural_innovations) ;
    StructuralShocksWeights = 1 ;
    I = 1 ;
    StructuralShocksMu = Q_lower_triangular_cholesky*(StructuralShocksMu') ;
    StructuralShocksSqrtP = zeros(number_of_structural_innovations,number_of_structural_innovations,I) ;
    StructuralShocksSqrtP(:,:,1) = Q_lower_triangular_cholesky ;
elseif ParticleOptions.mixture_structural_shocks==1
    if ParticleOptions.proposal_approximation.cubature
        [StructuralShocksMu,StructuralShocksWeights] = spherical_radial_sigma_points(number_of_structural_innovations);
        StructuralShocksWeights = ones(size(StructuralShocksMu,1),1)*StructuralShocksWeights ;
    elseif ParticleOptions.proposal_approximation.unscented
        [StructuralShocksMu,StructuralShocksWeights,raf] = unscented_sigma_points(number_of_structural_innovations,ParticleOptions);
    else
        if ~ParticleOptions.distribution_approximation.montecarlo
            error('Estimation: This approximation for the proposal is not implemented or unknown!')
        end
    end
    I = size(StructuralShocksWeights,1) ;
    StructuralShocksMu = Q_lower_triangular_cholesky*(StructuralShocksMu') ;
    StructuralShocksSqrtP = zeros(number_of_structural_innovations,number_of_structural_innovations,I) ;
    for i=1:I
        StructuralShocksSqrtP(:,:,i) = Q_lower_triangular_cholesky ;
    end
else
    if ParticleOptions.proposal_approximation.cubature
        [StructuralShocksMu,StructuralShocksWeights] = spherical_radial_sigma_points(number_of_structural_innovations);
        StructuralShocksWeights = ones(size(StructuralShocksMu,1),1)*StructuralShocksWeights ;
    elseif ParticleOptions.proposal_approximation.unscented
        [StructuralShocksMu,StructuralShocksWeights,raf] = unscented_sigma_points(number_of_structural_innovations,ParticleOptions);
    else
        if ~ParticleOptions.distribution_approximation.montecarlo
            error('Estimation: This approximation for the proposal is not implemented or unknown!')
        end
    end
    I = size(StructuralShocksWeights,1) ;
    StructuralShocksMu = Q_lower_triangular_cholesky*(StructuralShocksMu') ;
    StructuralShocksSqrtP = zeros(number_of_structural_innovations,number_of_structural_innovations,I) ;
    for i=1:I
        StructuralShocksSqrtP(:,:,i) = Q_lower_triangular_cholesky/sqrt(StructuralShocksWeights(i)) ;
    end
end

ObservationShocksMu = zeros(1,number_of_observed_variables) ;
ObservationShocksWeights = 1 ;
J = 1 ;
ObservationShocksMu = H_lower_triangular_cholesky*(ObservationShocksMu') ;
ObservationShocksSqrtP = zeros(number_of_observed_variables,number_of_observed_variables,J) ;
ObservationShocksSqrtP(:,:,1) = H_lower_triangular_cholesky ;

% if ParticleOptions.mixture_measurement_shocks==0
%     ObservationShocksMu = zeros(1,number_of_observed_variables) ;
%     ObservationShocksWeights = 1 ;
%     J = 1 ;
%     ObservationShocksMu = H_lower_triangular_cholesky*(ObservationShocksMu') ;
%     ObservationShocksSqrtP = zeros(number_of_observed_variables,number_of_observed_variables,J) ;
%     ObservationShocksSqrtP(:,:,1) = H_lower_triangular_cholesky ;
% elseif ParticleOptions.mixture_measurement_shocks==1
%     if ParticleOptions.proposal_approximation.cubature
%         [ObservationShocksMu,ObservationShocksWeights] = spherical_radial_sigma_points(number_of_observed_variables);
%         ObservationShocksWeights = ones(size(ObservationShocksMu,1),1)*ObservationShocksWeights;
%     elseif ParticleOptions.proposal_approximation.unscented
%         [ObservationShocksMu,ObservationShocksWeights,raf] = unscented_sigma_points(number_of_observed_variables,ParticleOptions);
%     else
%         if ~ParticleOptions.distribution_approximation.montecarlo
%             error('Estimation: This approximation for the proposal is not implemented or unknown!')
%         end
%     end
%     J = size(ObservationShocksWeights,1) ;
%     ObservationShocksMu = H_lower_triangular_cholesky*(ObservationShocksMu') ;
%     ObservationShocksSqrtP = zeros(number_of_observed_variables,number_of_observed_variables,J) ;
%     for j=1:J
%       ObservationShocksSqrtP(:,:,j) = H_lower_triangular_cholesky ;
%     end
% else
%     if ParticleOptions.proposal_approximation.cubature
%         [ObservationShocksMu,ObservationShocksWeights] = spherical_radial_sigma_points(number_of_observed_variables);
%         ObservationShocksWeights = ones(size(ObservationShocksMu,1),1)*ObservationShocksWeights;
%     elseif ParticleOptions.proposal_approximation.unscented
%         [ObservationShocksMu,ObservationShocksWeights,raf] = unscented_sigma_points(number_of_observed_variables,ParticleOptions);
%     else
%         if ~ParticleOptions.distribution_approximation.montecarlo
%             error('Estimation: This approximation for the proposal is not implemented or unknown!')
%         end
%     end
%     J = size(ObservationShocksWeights,1) ;
%     ObservationShocksMu = H_lower_triangular_cholesky*(ObservationShocksMu') ;
%     ObservationShocksSqrtP = zeros(number_of_observed_variables,number_of_observed_variables,J) ;
%     for j=1:J
%       ObservationShocksSqrtP(:,:,j) = H_lower_triangular_cholesky/sqrt(ObservationShocksWeights(j)) ;
%     end
% end

Gprime = G*I ;
Gsecond = G*I*J ;
SampleWeights = ones(Gsecond,1)/Gsecond ;

StateWeightsPrior = zeros(1,Gprime) ;
StateMuPrior = zeros(number_of_state_variables,Gprime) ;
StateSqrtPPrior = zeros(number_of_state_variables,number_of_state_variables,Gprime) ;

StateWeightsPost = zeros(1,Gsecond) ;
StateMuPost = zeros(number_of_state_variables,Gsecond) ;
StateSqrtPPost = zeros(number_of_state_variables,number_of_state_variables,Gsecond) ;

const_lik = (2*pi)^(.5*number_of_observed_variables) ;

lik  = NaN(sample_size,1);
LIK  = NaN;
for t=1:sample_size
    % Build the proposal joint quadratures of Gaussian on states, structural
    % shocks and observation shocks based on each combination of mixtures
    for i=1:I
        for j=1:J
            for g=1:G ;
                gprime = g + (i-1)*G ;
                gsecond = gprime + (j-1)*Gprime ;
                [StateMuPrior(:,gprime),StateSqrtPPrior(:,:,gprime),StateWeightsPrior(1,gprime),...
                 StateMuPost(:,gsecond),StateSqrtPPost(:,:,gsecond),StateWeightsPost(1,gsecond)] =...
                    gaussian_mixture_filter_bank(ReducedForm,Y(:,t),StateMu(:,g),StateSqrtP(:,:,g),StateWeights(g),...
                                                 StructuralShocksMu(:,i),StructuralShocksSqrtP(:,:,i),StructuralShocksWeights(i),...
                                                 ObservationShocksMu(:,j),ObservationShocksSqrtP(:,:,j),ObservationShocksWeights(j),...
                                                 H,H_lower_triangular_cholesky,const_lik,ParticleOptions,ThreadsOptions) ;
            end
        end
    end

    % Normalize weights
    StateWeightsPrior = StateWeightsPrior/sum(StateWeightsPrior,2) ;
    StateWeightsPost = StateWeightsPost/sum(StateWeightsPost,2) ;

    if ParticleOptions.distribution_approximation.cubature || ParticleOptions.distribution_approximation.unscented
        for i=1:Gsecond
            StateParticles = bsxfun(@plus,StateMuPost(:,i),StateSqrtPPost(:,:,i)*nodes') ;
            IncrementalWeights = gaussian_mixture_densities(Y(:,t),StateMuPrior,StateSqrtPPrior,StateWeightsPrior,...
                                                            StateMuPost,StateSqrtPPost,StateWeightsPost,...
                                                            StateParticles,H,const_lik,weights,weights_c,ReducedForm,ThreadsOptions) ;
            SampleWeights(i) = sum(StateWeightsPost(i)*weights.*IncrementalWeights) ;
        end
        SumSampleWeights = sum(SampleWeights) ;
        lik(t) = log(SumSampleWeights) ;
        SampleWeights = SampleWeights./SumSampleWeights ;
        [ras,SortedRandomIndx] = sort(rand(1,Gsecond));
        SortedRandomIndx = SortedRandomIndx(1:G);
        indx = resample(0,SampleWeights,ParticleOptions) ;
        indx = indx(SortedRandomIndx) ;
        StateMu = StateMuPost(:,indx);
        StateSqrtP = StateSqrtPPost(:,:,indx);
        StateWeights = ones(1,G)/G ;
    else
        % Sample particle in the proposal distribution, ie the posterior state GM
        StateParticles = importance_sampling(StateMuPost,StateSqrtPPost,StateWeightsPost',number_of_particles) ;
        IncrementalWeights = gaussian_mixture_densities(Y(:,t),StateMuPrior,StateSqrtPPrior,StateWeightsPrior,...
                                                        StateMuPost,StateSqrtPPost,StateWeightsPost,...
                                                        StateParticles,H,const_lik,1/number_of_particles,...
                                                        1/number_of_particles,ReducedForm,ThreadsOptions) ;
        SampleWeights = IncrementalWeights/number_of_particles ;
        SumSampleWeights = sum(SampleWeights,1) ;
        SampleWeights = SampleWeights./SumSampleWeights ;
        lik(t) = log(SumSampleWeights) ;
        if (ParticleOptions.resampling.status.generic && neff(SampleWeights)<ParticleOptions.resampling.threshold*sample_size) || ParticleOptions.resampling.status.systematic
            StateParticles = resample(StateParticles',SampleWeights',ParticleOptions)';
            SampleWeights = ones(number_of_particles,1)/number_of_particles;
        end
        [StateMu,StateSqrtP,StateWeights] = fit_gaussian_mixture(StateParticles,SampleWeights',StateMu,StateSqrtP,StateWeights,0.001,10,1) ;
    end
end

LIK = -sum(lik(start:end)) ;