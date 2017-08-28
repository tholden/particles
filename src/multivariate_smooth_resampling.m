function new_particles = multivariate_smooth_resampling(particles,weights)
% Smooth Resampling of the  particles.

%@info:
%! @deftypefn {Function File} {@var{new_particles} =} multivariate_smooth_resampling (@var{weights}, @var{particles}, @var{number_of_new_particles}, @var{number_of_partitions})
%! @anchor{particle/multivariate_smooth_resampling}
%! @sp 1
%! Smooth Resampling of the  particles (multivariate version).
%! @sp 2
%! @strong{Inputs}
%! @sp 1
%! @table @ @var
%! @item weights
%! n*1 vector of doubles, particles' weights.
%! @item particles
%! n*1 vector of doubles, particles.
%! @item number_of_new_particles
%! Integer scalar.
%! @item number_of_partitions
%! Integer scalar.
%! @end table
%! @sp 2
%! @strong{Outputs}
%! @sp 1
%! @table @ @var
%! @item indx
%! number_of_new_particles*1 vector of doubles, new particles.
%! @end table
%! @sp 2
%! @strong{This function is called by:}
%! @sp 1
%! @ref{particle/sequantial_importance_particle_filter}
%! @sp 2
%! @strong{This function calls:}
%! @sp 1
%! @ref{particle/univariate_smooth_resampling}
%! @sp 2
%! @end deftypefn
%@eod:

% Copyright (C) 2012-2017 Dynare Team
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

% AUTHOR(S) frederic DOT karame AT univ DASH lemans DOT fr
%           stephane DOT adjemian AT univ DASH lemans DOT fr

number_of_particles = length(weights);
number_of_states = size(particles,2);
[P,D] = eig(particles'*(bsxfun(@times,1/number_of_particles,particles))) ;
D = diag(D) ;
vectors = bsxfun(@times,P,sqrt(D)') ;
orthogonalized_particles = bsxfun(@rdivide,particles*vectors,D') ;
new_particles = zeros(number_of_particles,number_of_states) ;
for j=1:number_of_states
    tout = sortrows( [orthogonalized_particles(:,j) weights],1) ;
    new_particles(:,j) = univariate_smooth_resampling(tout(:,2),tout(:,1),number_of_particles) ;
end
new_particles = new_particles*(vectors') ;
