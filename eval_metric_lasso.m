function value = eval_metric_lasso(w, w_real, X, Y)
    % w: current estimator
    % W_real: ground truth
    % data: training data

    L2_norm = norm(w - w_real); % L2 norm
    L1_norm = norm(w - w_real, 1);
    w_ht = w .* (abs(w) > 0.01);

    S_ht = find(abs(w_ht) > 0);
    S = find(abs(w_real) > 0);

    precision = length(intersect(S_ht, S))/length(S_ht);
    recall = length(intersect(S_ht, S))/length(S);
    F1 = 2*precision*recall/(precision + recall);

    loss = norm(Y - X'*w)^2/length(Y);
    
    value = [L2_norm L1_norm F1 loss];
