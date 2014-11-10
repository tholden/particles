function measure = measurement_equations(StateVectors,ReducedForm,DynareOptions) 

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

mf1 = ReducedForm.mf1;
ghx  = ReducedForm.ghx(mf1,:);
ghu  = ReducedForm.ghu(mf1,:);
ghxx = ReducedForm.ghxx(mf1,:);
ghuu = ReducedForm.ghuu(mf1,:);
ghxu = ReducedForm.ghxu(mf1,:);
constant = ReducedForm.constant(mf1,:);
state_variables_steady_state = ReducedForm.state_variables_steady_state;
number_of_structural_innovations = length(ReducedForm.Q);
yhat = bsxfun(@minus,StateVectors,state_variables_steady_state) ;
measure = local_state_space_iteration_2(yhat,zeros(number_of_structural_innovations,size(yhat,2)),ghx,ghu,constant,ghxx,ghuu,ghxu,DynareOptions.threads.local_state_space_iteration_2);



