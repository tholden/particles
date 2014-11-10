function [c,SqrtVariance,Weights] = mykmeans(x,g,init,cod) 

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

[n,m] = size(x) ;
indold = zeros(1,m) ;
if cod==0
  d = transpose(sum(bsxfun(@power,bsxfun(@minus,x,mean(x)),2)));
  d = sortrows( [transpose(1:m) d],2) ;
  d = d((1+(0:1:g-1))*m/g,1) ;
  c = x(:,d);
else
  c = init ;
end 
for iter=1:300 
  dist = zeros(g,m) ;
  for i=1:g
    dist(i,:) = sum(bsxfun(@power,bsxfun(@minus,x,c(:,i)),2));
  end
  [rien,ind] = min(dist) ;
  if isequal(ind,indold) 
    break ;
  end
  indold = ind ;
  for i=1:g 
    lin = bsxfun(@eq,ind,i.*ones(1,m)) ;
    h = x(:,lin) ;
    c(:,i) = mean(h,2) ;
  end
end
SqrtVariance = zeros(n,n,g) ; 
Weights = zeros(1,g) ; 
for i=1:g
  temp = x(:,bsxfun(@eq,ind,i*ones(1,m))) ;
  u = bsxfun(@minus,temp,mean(temp,2)); %temp-mean(temp,1)' ;
  SqrtVariance(:,:,i) = chol( (u*u')/size(temp,2) )' ;
  Weights(i) = size(temp,2)/m ;
end
