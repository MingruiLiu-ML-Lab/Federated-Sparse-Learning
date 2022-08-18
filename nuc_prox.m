function A_st = nuc_prox(A, lambda)
    % soft thresholding
    % min_X  1/2 ||X - A||_2^2 +  lambda ||X||_nuc

    [U, S, V] = svd(A); % singular value decomposition for A;
    S_st = sign(S).*max(abs(S) - lambda, 0);
    A_st = U * S_st * V';
   
   
   