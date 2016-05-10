% Author:       Mixon, Villar, Ward.
% Filename:     get_data.m
% Last edited:  9 May 2016 
% Description:  This function loads data from './data/data_features.mat'.
%               The file was generated using a simple application of 
%               Tensor Flow [1] on the NMIST data set [2]. 
%               Contains two arrays: 
%               
%               -digits:
%               
%               A 10 x 1000 array. Each column of this array correspond to
%               a probability assignment of handwritten digit image onto
%               the 10 possible digits. 
%
%               -labels:
%               
%               A 10 x 1000 array corresponding to labels of the digits.
%               labels(i,n)=1 if the handwritten symbol n is the digit i-1,
%               and 0 otherwise.
%
% Inputs:       
% Outputs:
%               -digits:
%
%               Same as in description. The output digits are ordered by 
%               label.
%               
%               -labels:
%               
%               A 1000 x 1 array indicating the label for each digit.
% References:
% 
% [1] Abadi et al. TensorFlow: Large-scale machine learning on 
%       heterogeneous systems.
% [2] LeCun, Cortes. Mnist handwritten digit database.
% -------------------------------------------------------------------------

function [digits,labels]=get_data(FILENAME)

load(FILENAME,'digits', 'labels');
[m,N]=size(digits);

%sorting the digits according to labels
labels2=zeros(N,1);
for i=1:N
    [~, labels2(i,1)]= max(labels(:,i));
end
[num, indices]=sort(labels2);
digits=digits(:,indices);
labels=num;

end