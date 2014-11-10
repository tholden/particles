function [StateMu,StateSqrtP,StateWeights] = fit_gaussian_mixture(X,StateMu,StateSqrtP,StateWeights,crit,niters,check) 

% Copyright (C) 2013 Dynare Team
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

[dim,Ndata] = size(X);             
M = size(StateMu,2) ;
if check                        % Ensure that covariances don't collapse
  MIN_COVAR_SQRT = sqrt(eps);
  init_covars = StateSqrtP;
end
eold = -Inf;
for n=1:niters
  % Calculate posteriors based on old parameters
  [prior,likelihood,marginal,posterior] = probability(StateMu,StateSqrtP,StateWeights,X);
  e = sum(log(marginal));
  if (n > 1 && abs((e - eold)/eold) < crit)
    return;
  else
    eold = e;
  end
  new_pr = (sum(posterior,2))';
  StateWeights = new_pr/Ndata;
  StateMu = bsxfun(@rdivide,(posterior*X')',new_pr);
  for j=1:M
    diffs = bsxfun(@minus,X,StateMu(:,j));
    tpost = (1/sqrt(new_pr(j)))*sqrt(posterior(j,:));
    diffs = bsxfun(@times,diffs,tpost);
    [foo,tcov] = qr2(diffs',0);
    StateSqrtP(:,:,j) = tcov';
    if check
      if min(abs(diag(StateSqrtP(:,:,j)))) < MIN_COVAR_SQRT
        StateSqrtP(:,:,j) = init_covars(:,:,j);
      end
    end
  end
end     

