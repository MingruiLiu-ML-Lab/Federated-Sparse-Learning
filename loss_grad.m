function g = loss_grad(w,x,y)
   % d: dimension of feature
   % x: d-by-bs
   % y: bs-by-1
   % w: d-by-1
   
   pred = x'*w - y; % bs-by-1
      %g(ind) = pred*x';
   g = x*pred/size(pred,1);
end
