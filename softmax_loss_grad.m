function g = softmax_loss_grad(w,x,y)
    % w: (d, c)
    % x: (d, b)
    % y: (1, b)

    x_shape = size(x);
    d = x_shape(1);
    b = x_shape(2);
    w_shape = size(w);
    c = w_shape(2);

    logits = x'*w;
    probs = soft_max(logits);

    indicator = zeros(b, c);
    idxs = sub2ind(size(indicator), 1:b, y + 1); % offset k by 1 because labels in y range from 0 to 9
    indicator(idxs) = 1;
    diff = indicator - probs;
    diff = reshape(diff, 1, b, c);
    g = -1 * mean(x.*diff, 2);
    g = reshape(g, d, c);
end
