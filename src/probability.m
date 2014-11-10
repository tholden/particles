function [prior,likelihood,C,posterior] = probability(mu,sqrtP,prior,X)

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

[dim,nov] = size(X);              
M = size(mu,2) ;
if nargout>1
  likelihood = zeros(M,nov);        
  normfact = (2*pi)^(dim/2);  
  for k=1:M
    XX = bsxfun(@minus,X,mu(:,k));
    S = sqrtP(:,:,k);
    foo = S \ XX;
    likelihood(k,:) = exp(-0.5*sum(foo.*foo, 1))/abs((normfact*prod(diag(S))));
  end
end
likelihood = likelihood + 1e-99;
if nargout>2
  C = prior*likelihood + 1e-99;                   
end
if nargout>3
  posterior = bsxfun(@rdivide,bsxfun(@times,prior',likelihood),C) + 1e-99 ;
  posterior = bsxfun(@rdivide,posterior,sum(posterior,1));
end
