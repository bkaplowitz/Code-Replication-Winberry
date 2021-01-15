// Gives dynare equations governing model 
// From Winberry 2018

    // Conditional Expectation // 
@#for iEpsilon in 1 : nEpsilon
    @#for iAssets in 1 : nAssets
        // Compute the conditional expectation via chebyshev polynomials
        # expectation_@{iEpsilon}_@{iAssets} = exp(0 
        @#for iPower in 1 : nAssets
            + expectationCoefficient_@{iEpsilon}_@{iPower} * expectationPoly_@{iAssets}_@{iPower} 
        @#endfor
        );
        // Compute the savings policy a'(epsilon, a)
        # assetsPrime_@{iEpsilon}_@{iAssets} = max(w * (mu * (1 - epsilonGrid_@{iEpsilon}) + (1 - tau) * epsilonGrid_@{iEpsilon}) 
        + (1 + r) * assetsGrid_@{iAssets} - (expectation_@{iEpsilon}_@{iAssets} ^ (-1 / sigma) ), aBar);
        // Compute future consumption c'
        