// From Winberry, 2018
// Loads in each of the .mat data structs containing the parameters of interest

economicParameters = load('../Data/economicParameters.mat');
approximationParameters = load('../Data/approximationParameters.mat');
grids = load('../Data/grids.mat');
polynomials = load('../Data/polynomials.mat');
// Define the economic parameters
parameters beta sigma aBar aalpha delta N mu tau rhoTFP sigmaTFP;
// Load each of their values 
@#define nEconomicParameters = 10
for iParam = 1 : @{nEconomicParameters}
    // removes any spaces from name and accesses the parameter names stored in appropriate area
    parameterName = deblank(M_.param_names(iParam,:));
    // returns true if parametername is attribute in the .mat file given.
    // records the first parameter named as the object contained in the relevant .mat file

    if isfield(economicParameters,parameterName)
        M_.params(iParam) = eval(['economicParameters.' parameterName]);
    end
end 

// Reconstructs the epsilon transition matrix
// uses dynare loops 
//initializes a 2 by 2 epsilon transition matrix by creating four seperate variables linked to each entry
@#for iEpsilon in 1 : 2
    @#for iEpsilonPrime in 1 : 2
        parameters epsilonTransition_@{iEpsilon}_@{iEpsilonPrime};
    @#endfor
@#endfor
// assigns entries to 2x2 matrix from data 
// unrolls and accesses the right entries from the parameter matrix and assigns to each of the 4 created variables
for iEpsilon = 1 : 2
    for iEpsilonPrime = 1 : 2
        M_.params(@{nEconomicParameters} + 2 * (iEpsilon-1) +iEpsilonPrime) = economicParameters.matEpsilonTransition(iEpsilon,iEpsilonPrime);
    end
end

// Mass of invariant distribution of the idiosyncratic shocks
// Create the variables
parameters epsilonMass_1 epsilonMass_2; 
epsilonMass_1 = 1 - N; 
epsilonMass_2 = N; 
// Defines the set of approximation parameters 
parameters nEpsilon nAssets nState assetsMin assetsMax nAssetsFine nStateFine nAssetsQuad nStateQuad nMeasure nMeasureCoefficients kRepSS maxIterations tolerance dampening;
// Load in the values for each of them
@#define nApproximationParameters = 15
// goes through and assigns values from .mat to each of the parameters defined. Need +6 because the first 6 used as counters will have to be defined manually below. Also, starts after first 10 already defined above.
for iParam = 1: @{nApproximationParameters}
    parameterName = deblank(M_.param_names(@{nEconomicParameters} + 6 + iParam,:));
    //if isfield(approximationParameters, parameterName)
    M_.params(@{nEconomicParameters} + 6 + iParam) = eval(['approximationParameters.' parameterName]);
    //end 
end 
@#define nCounter = nEconomicParameters + 6 + nApproximationParameters

// need to explicitly tell dynare each of the right values of lengths (no len() functionality built in)
@#define nEpsilon = 2
@#define nAssets = 25
@#define nMeasure = 3
@#define nAssetsQuad = 8
@#define nState = nEpsilon * nAssets
@#define nStateQuad = nEpsilon * nAssetsQuad

    // Grids for approximating conditional expectation //

// Employment 

// Setup variables defined above
parameters epsilonGrid_1 epsilonGrid_2;

// Assigns values
epsilonGrid_1 = 0;
epsilonGrid_2 = 1;
//two new variables to account for 
@#define nCounter = nCounter + 2 

// Assets 

//define a new variable for each vector point subscripted by entry
@#for iAssets in 1 : nAssets
    parameters assetsGrid_@{iAssets};
@#endfor 
// Assign values (in same order that the parameters were declared hence why adjusting the counter)
for iAssets = 1 : @{nAssets}
    M_.params(@{nCounter} + iAssets) = grids.vecAssetsGrid(iAssets);
end

// Update the counter for the next set of parameter assignments 
@#define nCounter = nCounter + nAssets

    // Quad Grid and Weights // 

// Define parameters 
@#for iAssets in 1 : nAssetsQuad
    parameters quadGrid_@{iAssets};
    parameters quadWeights_@{iAssets};
@#endfor 

// Assign values 
for iAssets = 1 : @{nAssetsQuad}
    M_.params(@{nCounter} +2 * (iAssets - 1) + 1) = grids.vecAssetsGridQuad(iAssets);
    M_.params(@{nCounter} + 2* (iAssets - 1) + 2) = grids.vecQuadWeights(iAssets);
end

@#define nCounter = nCounter + 2 * nAssetsQuad

    // Conditional expectation polynomials //

// Chebyshev Polynomials 

// Defining parameters
@#for iAssets in 1 : nAssets
    @#for iPower in 1 : nAssets
        parameters expectationPoly_@{iAssets}_@{iPower};
    @#endfor 
@#endfor

// Assign values 
for iAssets = 1 : @{nAssets}
    for iPower = 1 : @{nAssets}
        M_.params(@{nCounter} + @{nAssets} * (iAssets - 1) + iPower) = polynomials.vecAssetsPoly(iAssets, iPower);
    end 
end 

// Update counter

@#define nCounter = nCounter + nAssets * nAssets

    // Squared terms of Chebyshev polynomials //

// Defining parameters 
@#for iAssets in 1: nAssets
    parameters expectationPolySquared_@{iAssets};
@#endfor 

// Assign the values 
for iAssets = 1 : @{nAssets}
    M_.params(@{nCounter} + iAssets) = polynomials.vecAssetsPolySquared(iAssets);
end

@#define nCounter = nCounter + nAssets

    // Quad grid poly // 

// Define params
@#for iAssets in 1 : nAssetsQuad
    @#for iPower in 1: nAssets 
        parameters quadPoly_@{iAssets}_@{iPower};
    @#endfor 
@#endfor 

// Assign values 
for iAssets = 1 : @{nAssetsQuad}
    for iPower = 1 : @{nAssets}
        M_.params(@{nCounter} + @{nAssets} * (iAssets - 1) + iPower) = polynomials.vecAssetsPolyQuad(iAssets,iPower);
    end
end

// update counter for future assignments

@#define nCounter = nCounter + nAssetsQuad * nAssets 

    // Borrowing constraint poly //

// Define parameters 
@#for iPower in 1 : nAssets 
    parameters bcPoly_@{iPower};
@#endfor 

// Assign values 
for iPower = 1 : @{nAssets}
    M_.params(@{nCounter} + iPower) = polynomials.vecAssetsPolyBC(iPower);
end 