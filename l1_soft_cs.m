function pr = l1_soft_cs(v, lambda, v0, rho)
    % soft thresholding
    % min_x  1/2 ||x - v||_2^2 +  lambda ||x||_1
    % s.t. ||x - v0||_1 <= rho

    pr = sign(v).*max(abs(v)-lambda,0);

    if sum(abs(pr - v0)) > rho
        pr = ProjectOntoL1Ball(pr - v0, rho);
        pr = pr + v0;
    end
