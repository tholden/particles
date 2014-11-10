function State_Particles = importance_sampling(StateMuPost,StateSqrtPPost,StateWeightsPost,numP)

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

[Xdim,Gsecond] = size(StateMuPost) ;  
u = rand(numP,1);
[Nc,comp] = histc(u, cumsum([0; StateWeightsPost]));    
State_Particles = zeros(Xdim,numP);
for k=1:Gsecond
  idx = bsxfun(@eq,comp,k*ones(size(comp))) ;
  State_Particles(:,idx) = StateSqrtPPost(:,:,k)*randn(Xdim,Nc(k));
end
State_Particles= State_Particles + StateMuPost(:,comp); 
    
