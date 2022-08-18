function loss = softmax_loss(w,x,y)
    % w: (d, c)
    % x: (d, b)
    % y: (1, b)

    shape = size(x);
    b = shape(2);

    logits = x'*w;
    probs = soft_max(logits);
    log_probs = log(probs);
    idxs = sub2ind(size(log_probs), 1:b, y + 1); % offset k by 1 because labels in y range from 0 to 9
    label_log_probs = log_probs(idxs);
    loss = -1 * mean(label_log_probs);
end
