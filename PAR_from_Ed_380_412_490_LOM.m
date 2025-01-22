function [PAR] = PAR_from_Ed_380_412_490_LOM(Ed)
%
% PAR predictor from multispectral Ed at the bands [380 412 490]
% Vectorized code, works with N samples at once, and returns N PAR
% estimates.
%
% PAR = PAR_from_Ed_380_412_490_LOM(Ed)
% Input arguments:
%   Ed · Nx3 matrix: downwelling irradiance spectrum
% Output arguments:
%   PAR    · Nx1 matrix: Estimated PAR
% where N is the number of samples.
%
% This function is self-contained.
% Jaime Pitarch, CNR-ISMAR, 27-Sep-2024.

% Example run:
% my_Ed =[0.00089262    0.0038237     0.018696
%         1.65e-06   1.0538e-05   0.00075004
%         0.058626      0.12932      0.14568];
%    PAR = PAR_from_Ed_380_443_490_555_v4_LOM(my_Ed);

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [-22.6422533091432;-20.6614926671289;-7.80877488738832];
x1_step1.gain = [0.0869916582169695;0.0939673350698428;0.23427495150328];
x1_step1.ymin = -1;

% Layer 1
b1 = [2.7780197875686689635;-3.0120016633103054993;-1.6388090036464972865;0.55657288715698627346;0.092343825930433970361;0.22503275975009690013;1.4326425339066723996;-1.6649349802087980432;2.1863399092884789887;3.2271132191366262987];
IW1_1 = [-2.3282050784214693984 1.9717753045713790616 -2.1704194149559303817;1.7702875398910571381 2.2668566031081960332 0.93870214854950884131;0.64605031248428668178 1.1124809461261702115 1.1106462750229051384;-0.44506226881435823062 0.97442965079100218873 2.3820454322470392405;2.7690038639963678868 1.2453509851470037084 1.1291872822199462423;-2.2050538340680700955 -1.8648169062170096844 0.23927310502370105483;1.9259010395722073739 2.0778392379817574032 1.1417721237266238443;-1.761174428590931873 -1.6946172080657826431 1.7279239861628628017;0.027955549285897381834 0.65680921655430168915 3.2126785914869842031;0.81947670564309249563 -2.5509019943299371391 -1.5022393076934437151];

% Layer 2
b2 = 0.87424730906479397596;
LW2_1 = [-1.9497831925718325241 -0.33267541788581661555 1.127219886179018582 0.27498930496927803668 0.47340814994358554335 0.83363927644876389955 0.72299360038495330549 0.4326085534077952488 -0.19726701849192673777 1.3298009128609376006];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 0.258205088375632;
y1_step1.xoffset = -3.99387529233967;

% ===== CALCULATIONS ========
x1=log10(Ed); % conversion to logarithmic form

% Dimensions
Q = size(x1,1); % samples

% Input 1
x1 = x1';
xp1 = mapminmax_apply(x1,x1_step1);

% Layer 1
a1 = tansig_apply(repmat(b1,1,Q) + IW1_1*xp1);

% Layer 2
a2 = repmat(b2,1,Q) + LW2_1*a1;

% Output 1
y1 = mapminmax_reverse(a2,y1_step1);
PAR = 10.^y1';
end

% ===== MODULE FUNCTIONS ========

% Map Minimum and Maximum Input Processing Function
function y = mapminmax_apply(x,settings)
y = bsxfun(@minus,x,settings.xoffset);
y = bsxfun(@times,y,settings.gain);
y = bsxfun(@plus,y,settings.ymin);
end

% Sigmoid Symmetric Transfer Function
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

% Map Minimum and Maximum Output Reverse-Processing Function
function x = mapminmax_reverse(y,settings)
x = bsxfun(@minus,y,settings.ymin);
x = bsxfun(@rdivide,x,settings.gain);
x = bsxfun(@plus,x,settings.xoffset);
end
