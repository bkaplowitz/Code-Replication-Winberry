function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
% function residual = dynamic_resid(T, y, x, params, steady_state, it_, T_flag)
%
% File created by Dynare Preprocessor from .mod file
%
% Inputs:
%   T             [#temp variables by 1]     double   vector of temporary terms to be filled by function
%   y             [#dynamic variables by 1]  double   vector of endogenous variables in the order stored
%                                                     in M_.lead_lag_incidence; see the Manual
%   x             [nperiods by M_.exo_nbr]   double   matrix of exogenous variables (in declaration order)
%                                                     for all simulation periods
%   steady_state  [M_.endo_nbr by 1]         double   vector of steady state values
%   params        [M_.param_nbr by 1]        double   vector of parameter values in declaration order
%   it_           scalar                     double   time period for exogenous variables for which
%                                                     to evaluate the model
%   T_flag        boolean                    boolean  flag saying whether or not to calculate temporary terms
%
% Output:
%   residual
%

if T_flag
    T = dynamics_polynomials.dynamic_resid_tt(T, y, x, params, steady_state, it_);
end
residual = zeros(71, 1);
lhs = log(T(2));
rhs = log(T(148));
residual(1) = lhs - rhs;
lhs = log(T(13));
rhs = log(T(158));
residual(2) = lhs - rhs;
lhs = log(T(14));
rhs = log(T(168));
residual(3) = lhs - rhs;
lhs = log(T(15));
rhs = log(T(178));
residual(4) = lhs - rhs;
lhs = log(T(16));
rhs = log(T(188));
residual(5) = lhs - rhs;
lhs = log(T(17));
rhs = log(T(198));
residual(6) = lhs - rhs;
lhs = log(T(18));
rhs = log(T(208));
residual(7) = lhs - rhs;
lhs = log(T(19));
rhs = log(T(218));
residual(8) = lhs - rhs;
lhs = log(T(20));
rhs = log(T(228));
residual(9) = lhs - rhs;
lhs = log(T(21));
rhs = log(T(238));
residual(10) = lhs - rhs;
lhs = log(T(22));
rhs = log(T(248));
residual(11) = lhs - rhs;
lhs = log(T(23));
rhs = log(T(258));
residual(12) = lhs - rhs;
lhs = log(T(24));
rhs = log(T(268));
residual(13) = lhs - rhs;
lhs = log(T(25));
rhs = log(T(278));
residual(14) = lhs - rhs;
lhs = log(T(26));
rhs = log(T(288));
residual(15) = lhs - rhs;
lhs = log(T(27));
rhs = log(T(298));
residual(16) = lhs - rhs;
lhs = log(T(28));
rhs = log(T(308));
residual(17) = lhs - rhs;
lhs = log(T(29));
rhs = log(T(318));
residual(18) = lhs - rhs;
lhs = log(T(30));
rhs = log(T(328));
residual(19) = lhs - rhs;
lhs = log(T(31));
rhs = log(T(338));
residual(20) = lhs - rhs;
lhs = log(T(32));
rhs = log(T(348));
residual(21) = lhs - rhs;
lhs = log(T(33));
rhs = log(T(358));
residual(22) = lhs - rhs;
lhs = log(T(34));
rhs = log(T(368));
residual(23) = lhs - rhs;
lhs = log(T(35));
rhs = log(T(378));
residual(24) = lhs - rhs;
lhs = log(T(36));
rhs = log(T(388));
residual(25) = lhs - rhs;
lhs = log(T(37));
rhs = log(T(398));
residual(26) = lhs - rhs;
lhs = log(T(39));
rhs = log(T(408));
residual(27) = lhs - rhs;
lhs = log(T(40));
rhs = log(T(418));
residual(28) = lhs - rhs;
lhs = log(T(41));
rhs = log(T(428));
residual(29) = lhs - rhs;
lhs = log(T(42));
rhs = log(T(438));
residual(30) = lhs - rhs;
lhs = log(T(43));
rhs = log(T(448));
residual(31) = lhs - rhs;
lhs = log(T(44));
rhs = log(T(458));
residual(32) = lhs - rhs;
lhs = log(T(45));
rhs = log(T(468));
residual(33) = lhs - rhs;
lhs = log(T(46));
rhs = log(T(478));
residual(34) = lhs - rhs;
lhs = log(T(47));
rhs = log(T(488));
residual(35) = lhs - rhs;
lhs = log(T(48));
rhs = log(T(498));
residual(36) = lhs - rhs;
lhs = log(T(49));
rhs = log(T(508));
residual(37) = lhs - rhs;
lhs = log(T(50));
rhs = log(T(518));
residual(38) = lhs - rhs;
lhs = log(T(51));
rhs = log(T(528));
residual(39) = lhs - rhs;
lhs = log(T(52));
rhs = log(T(538));
residual(40) = lhs - rhs;
lhs = log(T(53));
rhs = log(T(548));
residual(41) = lhs - rhs;
lhs = log(T(54));
rhs = log(T(558));
residual(42) = lhs - rhs;
lhs = log(T(55));
rhs = log(T(568));
residual(43) = lhs - rhs;
lhs = log(T(56));
rhs = log(T(578));
residual(44) = lhs - rhs;
lhs = log(T(57));
rhs = log(T(588));
residual(45) = lhs - rhs;
lhs = log(T(58));
rhs = log(T(598));
residual(46) = lhs - rhs;
lhs = log(T(59));
rhs = log(T(608));
residual(47) = lhs - rhs;
lhs = log(T(60));
rhs = log(T(618));
residual(48) = lhs - rhs;
lhs = log(T(61));
rhs = log(T(628));
residual(49) = lhs - rhs;
lhs = log(T(62));
rhs = log(T(638));
residual(50) = lhs - rhs;
lhs = y(1);
rhs = T(639)/T(648);
residual(51) = lhs - rhs;
lhs = y(3);
rhs = T(649)/T(648);
residual(52) = lhs - rhs;
lhs = y(5);
rhs = T(650)/T(648);
residual(53) = lhs - rhs;
lhs = y(2);
rhs = T(651)/T(660);
residual(54) = lhs - rhs;
lhs = y(4);
rhs = T(661)/T(660);
residual(55) = lhs - rhs;
lhs = y(6);
rhs = T(662)/T(660);
residual(56) = lhs - rhs;
lhs = y(60);
rhs = (T(129)*T(679)/T(648)+T(130)*T(681)+T(131)*T(698)/T(660)+T(132)*T(700))/params(15);
residual(57) = lhs - rhs;
lhs = y(62);
rhs = (T(709)/T(648)+T(130)*T(710)+T(719)/T(660)+T(132)*T(720))/params(15);
residual(58) = lhs - rhs;
lhs = y(64);
rhs = (T(729)/T(648)+T(130)*T(730)+T(739)/T(660)+T(132)*T(740))/params(15);
residual(59) = lhs - rhs;
lhs = y(61);
rhs = (T(133)*T(679)/T(648)+T(134)*T(681)+T(135)*T(698)/T(660)+T(136)*T(700))/params(16);
residual(60) = lhs - rhs;
lhs = y(63);
rhs = (T(749)/T(648)+T(134)*T(750)+T(759)/T(660)+T(136)*T(760))/params(16);
residual(61) = lhs - rhs;
lhs = y(65);
rhs = (T(769)/T(648)+T(134)*T(770)+T(779)/T(660)+T(136)*T(780))/params(16);
residual(62) = lhs - rhs;
lhs = y(72);
rhs = (T(129)*T(781)/T(648)+T(130)*(T(681)<=params(3)+1e-8)+T(131)*T(782)/T(660)+T(132)*(T(700)<=params(3)+1e-8))/params(15);
residual(63) = lhs - rhs;
lhs = y(73);
rhs = (T(133)*T(781)/T(648)+T(134)*(T(681)<=params(3)+1e-8)+T(135)*T(782)/T(660)+T(136)*(T(700)<=params(3)+1e-8))/params(16);
residual(64) = lhs - rhs;
lhs = y(74);
rhs = T(783)-params(5);
residual(65) = lhs - rhs;
lhs = y(75);
rhs = T(138)*exp(y(76))*(1-params(4))*T(784);
residual(66) = lhs - rhs;
lhs = y(76);
rhs = params(9)*y(9)+params(10)*x(it_, 1);
residual(67) = lhs - rhs;
lhs = y(77);
rhs = log(T(137)*exp(y(76))*T(784));
residual(68) = lhs - rhs;
lhs = y(78);
rhs = log(T(785));
residual(69) = lhs - rhs;
lhs = y(79);
rhs = log(exp(y(77))-exp(y(78)));
residual(70) = lhs - rhs;
lhs = y(80);
rhs = log(y(75));
residual(71) = lhs - rhs;

end
