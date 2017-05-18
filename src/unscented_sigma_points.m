function [nodes,W_m,W_c] = unscented_sigma_points(n,ParticleOptions)
%
% Computes nodes and weigths for a scaled unscented transform cubature
% INPUTS
%    n                  [integer]   scalar, number of variables.
%
% OUTPUTS
%    nodes          [double]    nodes of the cubature
%    weigths        [double]    associated weigths
%
% REFERENCES
%
%
%
% NOTES
%
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

lambda = (ParticleOptions.unscented.alpha^2)*(n+ParticleOptions.unscented.kappa) - n ;
nodes = [ zeros(n,1) ( sqrt(n+lambda).*([ eye(n) -eye(n)]) ) ]' ;
W_m = lambda/(n+lambda) ;
W_c = W_m + (1-ParticleOptions.unscented.alpha^2+ParticleOptions.unscented.beta) ;
temp = ones(2*n,1)/(2*(n+lambda)) ;
W_m = [W_m ; temp] ;
W_c = [W_c ; temp]  ;
