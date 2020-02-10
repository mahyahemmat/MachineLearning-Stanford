function [err_train, err_val] = randerror(itr, X_poly, y, X_poly_val,yval,lambda)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
for k=1:size(X_poly,1)
err_train=zeros(k,itr);
err_val=zeros(k,itr);
%
for j=1:itr
    rand_indice=randperm( size(X_poly,1),k); %num_example integers between 1 to size(x_poly,1)
    X_poly_new=X_poly(rand_indice,:);
    y_new=y(rand_indice);
    
    rand_indice_val=randperm( size(X_poly_val,1),k);
    X_poly_val_new=X_poly_val(rand_indice_val,:);
    yval_new=yval(rand_indice_val);
    
    [theta] = trainLinearReg(X_poly_new, y_new, lambda);
    [J, grad] = linearRegCostFunction(X_poly_new, y_new, theta, 0);
    error_train(j,k)=J;
    [J, grad] = linearRegCostFunction(X_poly_val_new, yval_new, theta, 0);
    error_val(j,k)=J;
   
end

end

err_train=mean(error_train,1);
err_val=mean(error_val,1);