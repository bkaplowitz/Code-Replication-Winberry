// First Order Dynamics
// Based on Winberry, 2018
    //Load parameters //
@#include "parameters_polynomials.mod"
    // Variable Definition //
\\ loads variables stored here
@#include "variables_polynomials.mod"
    // Model Equations //
 \\ loads equations stored here
model; 
    @#include "equations_polynomials.mod"
end;
    // Computation //
// Specify the shock process 
 shocks;
    var aggregateTFPShock = 1;
 end;

 options_.steadystate.nocheck = 1;
 // Computes the steady state while avoiding checking for small numerical errors

    // Check regularity conditions (optional) //
 // check;
 // model_diagnostics;
 // model_info; 

    // Simulate //
 stoch_simul(order=1, hp_filter=100,irf=40) aggregateTFP logAggregateOutput logAggregateConsumption logAggregateInvestment logWage r;