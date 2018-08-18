function [ ix2, temperedlogpost, loglik, ModelName, MetropolisFolder, npar, NumberOfParticles, bayestopt_] = ...
    DSMH_initialization(TargetFun, xparam1, mh_bounds,dataset_,dataset_info,options_,M_,estim_params_,bayestopt_,oo_)
% function [ ix2, ilogpo2, ModelName, MetropolisFolder, FirstBlock, FirstLine, npar, NumberOfParticles, bayestopt_] = ...
%     DSMH_initialization(TargetFun, xparam1, mh_bounds,dataset_,dataset_info,options_,M_,estim_params_,bayestopt_,oo_)
% Dynamic Striated Metropolis-Hastings initialization.
%
% INPUTS
%   o TargetFun  [char]     string specifying the name of the objective
%                           function (tempered posterior kernel and likelihood).
%   o xparam1    [double]   (p*1) vector of parameters to be estimated (initial values).
%   o mh_bounds  [double]   (p*2) matrix defining lower and upper bounds for the parameters.
%   o dataset_              data structure
%   o dataset_info          dataset info structure
%   o options_              options structure
%   o M_                    model structure
%   o estim_params_         estimated parameters structure
%   o bayestopt_            estimation options structure
%   o oo_                   outputs structure
%
% OUTPUTS
%   o ix2                   [double]    (NumberOfParticles*npar) vector of starting points for different chains
%   o ilogpo2               [double]    (NumberOfParticles*1) vector of initial posterior values for different chains
%   o iloglik2              [double]    (NumberOfParticles*1) vector of initial likelihood values for different chains
%   o ModelName             [string]    name of the mod-file
%   o MetropolisFolder      [string]    path to the Metropolis subfolder
%   o npar                  [scalar]    number of parameters estimated
%   o NumberOfParticles     [scalar]    Number of particles requested for the parameters distributions
%   o bayestopt_            [structure] estimation options structure
%
% SPECIAL REQUIREMENTS
%   None.

% Copyright (C) 2006-2017 Dynare Team
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

%Initialize outputs
ix2 = [];
ilogpo2 = [];
iloglik2 = [];
ModelName = [];
MetropolisFolder = [];
npar = [];
NumberOfParticles = [];

ModelName = M_.fname;
if ~isempty(M_.bvar)
    ModelName = [ModelName '_bvar'];
end

MetropolisFolder = CheckPath('dsmh',M_.dname);
BaseName = [MetropolisFolder filesep ModelName];

NumberOfParticles = options_.dsmh.number_of_particles; %Number of particles for the parameters
npar  = length(xparam1);

% Here we start a new DS Metropolis-Hastings, previous draws are discarded.
disp('Estimation::dsmh: Initialization...')
% Delete old dsmh files if any...
files = dir([BaseName '_dsmh*_blck*.mat']);
%if length(files)
%    delete([BaseName '_dsmh*_blck*.mat']);
%    disp('Estimation::smc: Old dsmh-files successfully erased!')
%end
% Delete old log file.
file = dir([ MetropolisFolder '/dsmh.log']);
%if length(file)
%    delete([ MetropolisFolder '/dsmh.log']);
%    disp('Estimation::dsmh: Old dsmh.log file successfully erased!')
%    disp('Estimation::dsmh: Creation of a new dsmh.log file.')
%end
fidlog = fopen([MetropolisFolder '/dsmh.log'],'w');
fprintf(fidlog,'%% DSMH log file (Dynare).\n');
fprintf(fidlog,['%% ' datestr(now,0) '.\n']);
fprintf(fidlog,' \n\n');
fprintf(fidlog,'%% Session 1.\n');
fprintf(fidlog,' \n');
prior_draw(bayestopt_,options_.prior_trunc);
% Find initial values for the NumberOfParticles chains...
set_dynare_seed('default');
fprintf(fidlog,['  Initial values of the parameters:\n']);
disp('Estimation::dsmh: Searching for initial values...');
ix2 = zeros(npar,NumberOfParticles);
temperedlogpost = zeros(NumberOfParticles,1);
loglik = zeros(NumberOfParticles,1);
%stderr = sqrt(bsxfun(@power,mh_bounds.ub-mh_bounds.lb,2)/12)/10;
for j=1:NumberOfParticles
    validate = 0;
    while validate == 0
    	candidate = prior_draw()';
%        candidate = xparam1(:) + 0.001*randn(npar,1);%bsxfun(@times,stderr,randn(npar,1)) ;
        if all(candidate(:) >= mh_bounds.lb) && all(candidate(:) <= mh_bounds.ub)
            ix2(:,j) = candidate ;
            [temperedlogpost(j),loglik(j)] = tempered_likelihood(TargetFun,candidate,0.0,dataset_,dataset_info,options_,M_,estim_params_,bayestopt_,mh_bounds,oo_);
            if isfinite(loglik(j)) % if returned log-density is Inf or Nan (penalized value)
                validate = 1;
            end
        end
    end
end
fprintf(fidlog,' \n');
disp('Estimation::dsmh: Initial values found!')
skipline()


