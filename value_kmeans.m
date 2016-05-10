% Author:       Mixon, Villar, Ward.
% Filename:     value_kmeans.m
% Last edited:  9 May 2016 
% Description:  This function computes the kmeans value (sum of squared
%               mean deviations from cluster centroid) of a 
%               provided partition of a provided set of points.
%               
%
% Inputs:       -points:
%               
%               A d x N array. Each column of this array correspond the
%               coordinates of a data point. 
%
%               -labels:
%               
%               A N x 1 array corresponding to the assignments of the 
%               points in k clusters.
%               
% Outputs:
%               -value:
%               
%               The kmeans value for the provided assignment.
% 
% References:
% 
% -------------------------------------------------------------------------

function value= value_kmeans(points, labels)

k=max(labels);
points=points';
[assign, order]=sort(labels);
points=points(order,:);

count=zeros(k,1);
for i=1:k
    count(i)=sum(assign==i);
end

idx=0;
value=0;
for t=1:k
    cluster=points(idx+1:idx+count(t),:);
    center=ones(1, size(cluster,1))*cluster/count(t);
    for i=1:count(t)
        value=value+norm(cluster(i,:)- center)^2;
    end
    idx=idx+count(t);
end

end