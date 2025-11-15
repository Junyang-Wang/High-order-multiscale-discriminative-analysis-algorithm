function [r]=corr3(a,b)
% computes the correlation coefficient between tensor A and tensor B,
% where A,B are 3D Tensors of the same size.
Na = ndims(a);
Nb = ndims(b);
if Na==Nb
   N = Na;
   a = a - mean2(a);
   b = b - mean2(b);
   r = ttt(tensor(a),tensor(b),1:N)/sqrt(ttt(tensor(a),tensor(a),1:N)*ttt(tensor(b),tensor(b),1:N)); 

else 
    error('the dimension of input data error!')
end
end