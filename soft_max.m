function soft = soft_max(a)
    shape = size(a);
    c = shape(2);
    expon = exp(a);
    denom = sum(expon, 2);
    denom = repmat(denom, 1, c);
    soft = expon./denom;
end
