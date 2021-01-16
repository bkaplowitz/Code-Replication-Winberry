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
        + (1 + r) * assetsGrid_@{iAssets} - (expectation_@{iEpsilon}_@{iAssets} ^(-1 / sigma)), aBar);
        // Compute future consumption c'
        @#for iEpsilonPrime in 1 : nEpsilon
            // future EMUC approximation

            # expectationPrime_@{iEpsilonPrime}_@{iEpsilon}_@{iAssets} = exp(0
            @#for iPower in 1 : nAssets
                + expectationCoefficient_@{iEpsilonPrime}_@{iPower}(+1) * cos((@{iPower} - 1) * acos(min(max(
                    2 * ((assetsPrime_@{iEpsilon}_@{iAssets} - assetsMin)/(assetsMax - assetsMin)) - 1,-1),1)))
            @#endfor
            );

            // compute a'' given approximation for EMUC using budget constraint + EE.
            # assetsPrime_@{iEpsilonPrime}_@{iEpsilon}_@{iAssets} = max(w(+1) * (mu * (1 - epsilonGrid_@{iEpsilonPrime}) + 
				(1 - tau) * epsilonGrid_@{iEpsilonPrime}) + (1 + r(+1)) * assetsPrime_@{iEpsilon}_@{iAssets} - 
				(expectationPrime_@{iEpsilonPrime}_@{iEpsilon}_@{iAssets} ^ (-1 / sigma)),aBar);
            
            // compute c' given a' and a'' 
            # consumptionPrime_@{iEpsilonPrime}_@{iEpsilon}_@{iAssets} = w(+1) * (mu * (1 - epsilonGrid_@{iEpsilonPrime}) + (1 - tau) * epsilonGrid_@{iEpsilonPrime}) +
            (1 + r(+1)) * assetsPrime_@{iEpsilon}_@{iAssets} - assetsPrime_@{iEpsilonPrime}_@{iEpsilon}_@{iAssets};

        @#endfor

        // Functional equation for EMUC used in compute_MC_Residual_Poly.py
        log(expectation_@{iEpsilon}_@{iAssets}) = log(beta * (1 + r(+1)) * (0 
        @#for iEpsilonPrime in 1 : nEpsilon
            + epsilonTransition_@{iEpsilon}_@{iEpsilonPrime} * (consumptionPrime_@{iEpsilonPrime}_@{iEpsilon}_@{iAssets} ^(-sigma))
        @#endfor
        ));

    @#endfor
@#endfor

    // Compute various objects over quad grid for integrating distribution //

@#for iEpsilon in 1 : nEpsilon
    
    @#for iAssets in 1 : nAssetsQuad
        //Compute conditional expectation for quadrature
        # expectationQuad_@{iEpsilon}_@{iAssets} = exp(0
        @#for iPower in 1 : nAssets 
            + expectationCoefficient_@{iEpsilon}_@{iPower} * quadPoly_@{iAssets}_@{iPower}
        @#endfor
        );
        // Compute savings policy
        # assetsPrimeQuad_@{iEpsilon}_@{iAssets} = max(w*(mu* (1-epsilonGrid_@{iEpsilon}) + 
        (1-tau)* epsilonGrid_@{iEpsilon}) + (1+r) * quadGrid_@{iAssets} - (expectationQuad_@{iEpsilon}_@{iAssets}^(-1 / sigma)),aBar);

        // density of the distribution-- uses last estimate from iteration to estimate next pdf and then generate moments
        # measurePDF_@{iEpsilon}_@{iAssets} = exp( 0 + measureCoefficient_@{iEpsilon}_1 * 
        (quadGrid_@{iAssets} - moment_@{iEpsilon}_1(-1)) 
        @#for iMoment in 2 : nMeasure
            + measureCoefficient_@{iEpsilon}_@{iMoment}*((quadGrid_@{iAssets} - moment_@{iEpsilon}_1(-1))^@{iMoment}-
            moment_@{iEpsilon}_@{iMoment}(-1))
        @#endfor
        );
    
    @#endfor

    // Total Mass of the distribution

    # totalMass_@{iEpsilon} = (0 
    @#for iAssets in 1 : nAssetsQuad 
        + quadWeights_@{iAssets} * measurePDF_@{iEpsilon}_@{iAssets} 
    @#endfor
    );

@#endfor 

    // Compute economic equations at borrowing constraint for later integration //

@#for iEpsilon in 1 : nEpsilon
    # expectationBC_@{iEpsilon} = exp(0 
    @#for iPower in 1 : nAssets
        + expectationCoefficient_@{iEpsilon}_@{iPower} * bcPoly_@{iPower}
    @#endfor
    );

    // Compute optimal savings policy at BC
    # assetsPrimeBC_@{iEpsilon} = max(w*(mu*(1-epsilonGrid_@{iEpsilon}) + (1-tau)* epsilonGrid_@{iEpsilon}) +
    (1+r)*aBar - (expectationBC_@{iEpsilon}^(-1/sigma)),aBar);
@#endfor 

    // Relationship between moments of the distribution and the parameters (integral equations) //

@#for iEpsilon in 1 : nEpsilon
    
    // First moments (uncentered)
    moment_@{iEpsilon}_1(-1) = (0
    @#for iAssets in 1 : nAssetsQuad 
        + quadWeights_@{iAssets}*quadGrid_@{iAssets}*measurePDF_@{iEpsilon}_@{iAssets}
    @#endfor
    ) / totalMass_@{iEpsilon};

    // Higher order moments (centered)
    @#for iMoment in 2 : nMeasure
        moment_@{iEpsilon}_@{iMoment}(-1) = (0
        @#for iAssets in 1 : nAssetsQuad
            + quadWeights_@{iAssets} * measurePDF_@{iEpsilon}_@{iAssets} * 
            ((quadGrid_@{iAssets} - moment_@{iEpsilon}_1(-1))^@{iMoment})
        @#endfor
        ) / totalMass_@{iEpsilon};
    @#endfor 
@#endfor
// Law of motion for density away from the borrowing constraint //
@#for iEpsilonPrime in 1 : nEpsilon

    // First moment (uncentered)
    moment_@{iEpsilonPrime}_1 = (0
    @#for iEpsilon in 1 : nEpsilon
        + ((1- mHat_@{iEpsilon}(-1)) * epsilonMass_@{iEpsilon} * 
        epsilonTransition_@{iEpsilon}_@{iEpsilonPrime} * (0
        @#for iAssets in 1 : nAssetsQuad
            + quadWeights_@{iAssets} * measurePDF_@{iEpsilon}_@{iAssets} * assetsPrimeQuad_@{iEpsilon}_@{iAssets}
        @#endfor
        ) / totalMass_@{iEpsilon}) + mHat_@{iEpsilon}(-1) * epsilonMass_@{iEpsilon} *
        epsilonTransition_@{iEpsilon}_@{iEpsilonPrime} * assetsPrimeBC_@{iEpsilon}
    @#endfor
    ) / epsilonMass_@{iEpsilonPrime};

    // Higher order moments (uncentered)
    @#for iMoment in 2 : nMeasure
        moment_@{iEpsilon}_@{iMoment} = (0
        @#for iEpsilon in 1 : nEpsilon
            + ((1- mHat_@{iEpsilon}(-1)) * epsilonMass_@{iEpsilon} * epsilonTransition_@{iEpsilon}_@{iEpsilonPrime} *(0
            @#for iAssets in 1 : nAssetsQuad
            + quadWeights_@{iAssets} * measurePDF_@{iEpsilon}_@{iAssets} * 
            (assetsPrimeQuad_@{iEpsilon}_@{iAssets} - moment_@{iEpsilonPrime}_1)^@{iMoment}
            @#endfor 
            ) / totalMass_@{iEpsilon} ) + mHat_@{iEpsilon}(-1) * epsilonMass_@{iEpsilon} * 
            epsilonTransition_@{iEpsilon}_@{iEpsilonPrime} * (assetsPrimeBC_@{iEpsilon} - moment_@{iEpsilonPrime}_1)^@{iMoment}
        @#endfor
        ) / epsilonMass_@{iEpsilonPrime};
    @#endfor

@#endfor

    // Law of motion for mass at borrowing constraints //

@#for iEpsilonPrime in 1 : nEpsilon

    mHat_@{iEpsilonPrime} = (0
    @#for iEpsilon in 1 : nEpsilon
        + ((1 - mHat_@{iEpsilon}(-1)) * epsilonMass_@{iEpsilon} * epsilonTransition_@{iEpsilon}_@{iEpsilonPrime} * (0
        @#for iAssets in 1 : nAssetsQuad
            + quadWeights_@{iAssets} * measurePDF_@{iEpsilon}_@{iAssets} * 
            (assetsPrimeQuad_@{iEpsilon}_@{iAssets}<= aBar + 1e-8)

        @#endfor
        ) / totalMass_@{iEpsilon}) + mHat_@{iEpsilon}(-1) * epsilonMass_@{iEpsilon} * 
        epsilonTransition_@{iEpsilon}_@{iEpsilonPrime} * (assetsPrimeBC_@{iEpsilon} <= aBar + 1e-8)
    @#endfor
    ) / epsilonMass_@{iEpsilonPrime};

@#endfor

    // Factor Prices //

// From law of iterated expectations, removing conditional on state of employment from mean

# aggregateCapital = (1-N) * moment_1_1(-1) + N * moment_2_1(-1);

r = exp(aggregateTFP) * alpha * (aggregateCapital ^ (alpha - 1)) * (N ^ (1-alpha)) - delta;
w = exp(aggregateTFP) * (1-alpha) * (aggregateCapital ^ alpha) * (N ^ (-alpha));

    // Law of Motion for Aggregate TFP //

aggregateTFP = rhoTFP * aggregateTFP(-1) + sigmaTFP * aggregateTFPShock;

    // Auxiliary variables we want to output//

// Log Output 
logAggregateOutput = log(exp(aggregateTFP) * (aggregateCapital ^ alpha) * (N ^(1 - alpha)));

// Log Investment
logAggregateInvestment = log((1-N) * moment_1_1 + N * moment_2_1 - (1-delta) * aggregateCapital);

// Log Consumption
logAggregateConsumption = log(exp(logAggregateOutput) - exp(logAggregateInvestment));

// Wage
logWage = log(w);


        


