// Tells dynare what the variables are for dynamics.mod 
// Taken from Winberry, 2018

    // Conditional Expectation coefficients //

@#for iPower in 1 : nAssets
    var expectationCoefficient_1_@{iPower} expectationCoefficient_2_@{iPower};
@#endfor 

    // Density of HH outside of BC //

// Record each moment of distribution up to nMeasure
@#for iMoment in 1:nMeasure 
    var moment_1_@{iMoment} moment_2_@{iMoment};
@#endfor 

// Parameters of the distribution
@#for iParameter in 1 : nMeasure
    var measureCoefficient_1_@{iParameter} measureCoefficient_2_@{iParameter};
@#endfor

    // Mass at the borrowing constraint //

var mHat_1 mHat_2;

    // Prices // 

var r w;
    // Aggregate TFP //

var AggregateTFP;

    // Auxiliary variables we want to know // 
    
var logAggregateOutput logAggregateInvestment logAggregateConsumption logWage;

    // Shocks //

varexo aggregateTFPShock;
