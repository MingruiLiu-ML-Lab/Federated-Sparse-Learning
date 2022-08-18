function A_st = nuc_prox_cs(A, lambda, W0, rho)
    % min_X  1/2 ||X - A||_2^2 +  lambda ||X||_nuc
    % s.t.   ||X - W0||_nuc <= rho

    [U, S, V] = svd(A); % singular value decomposition for A;
    S_st = sign(S).*max(abs(S) - lambda, 0);
    A_st = U * S_st * V';

    [Uv, Sv, Vv] = svd(A_st - W0);
    if sum(Sv) > rho
        S_proj = ProjectOntoSimplex(Sv, rho);
        A_st = Uv * S_proj * Vv + W0;
    end



