function pr = l1_soft(v,lambda)
 % soft thresholding
 % min_x  1/2 ||x - v||_2^2 +  lambda ||x||_1
 
  %pr = abs(v)-lambda;
  %bool = pr > 0;
  %pr(~bool) = 0;
  %pr(bool) = sign(v(bool)).*pr(bool);  

  pr = sign(v).*max(abs(v)-lambda,0);
