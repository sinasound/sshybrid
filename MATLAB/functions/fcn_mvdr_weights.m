function w = fcn_mvdr_weights(R,steer_vec)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
invRd = R \ steer_vec;
w = invRd / (steer_vec' * invRd);
end