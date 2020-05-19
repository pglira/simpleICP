function [idx, dist] = knnsearch(X, Y, k)

  D = pdist2(X, Y, "euclidean");

  [DSorted, idxSorted] = sort(D);

  idx = transpose(idxSorted(1:k, :));

  dist = transpose(DSorted(1:k, :));

endfunction
