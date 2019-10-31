function out = softmax(x)
    out = exp(x)./ sum(exp(x));
end