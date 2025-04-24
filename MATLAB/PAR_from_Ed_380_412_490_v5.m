function [PAR,PAR_b,ep50,IQR_ep] = PAR_from_Ed_380_412_490_v5(Ed,z)
%
% PAR predictor from multispectral Ed at the bands [380 412 490]
% Vectorized code, works with N samples at once, and returns N PAR
% estimates.
%
% [PAR,PAR_b,ep50,IQR_ep] = PAR_from_Ed_380_412_490_v5(Ed,z)
% Input arguments:
%   Ed · Nx3 matrix: downwelling irradiance spectrum
%   z ·  Nx1 matrix: depth (m)
% Output arguments:
%   PAR    · Nx1 matrix: Best PAR estimate
%   PAR_b  · Nx1 matrix: PAR with some remaining biases to be corrected with ep50
%   ep50   · Nx1 matrix: Estimated median percent error of the output value, as a function of depth
%   IQR_ep · Nx1 matrix: Estimated interquartile range of the percent error of the output value, as a function of depth
% where N is the number of samples.
%
% This function is self-contained.
% Jaime Pitarch, CNR-ISMAR, 08-May-2025.

% Example run:
% my_Ed =[   0.00026646    0.0017325      0.01242
%     0.0040838     0.017985     0.027024
%     3.928e-05   0.00033364    0.0033945];
%
% z=[      43.1
%         119.7
%         137.8];
%    [PAR,PAR_b,ep50,IQR_ep] = PAR_from_Ed_380_443_490_560_v5(Ed,z);

% ===== NEURAL NETWORK CONSTANTS =====

% Input 1
x1_step1.xoffset = [-22.6422533091432;-20.6614926671289;-7.62188447185006];
x1_step1.gain = [0.0862100486323673;0.0930530950102774;0.233748143539256];
x1_step1.ymin = -1;

% Layer 1
b1 = [2.3534783362386342276;-2.7662240071825312526;-1.2023536811814286018;1.2040903362406929489;-0.075233556686348096454;-0.11505354082018982853;1.5057877379280761865;-1.6671151010181339824;2.3498103376541426002;3.2094164560983413637];
IW1_1 = [-2.1843026643493397287 2.4145037279221663873 -2.4653281155219315401;1.6936390019822271658 2.3240464544790890855 0.92138063974384132315;0.39801299584177285418 1.0347975632886470265 1.0124844033661655196;-0.05665455451357419292 0.12498771361002697367 2.6057100732154752087;2.4276036213288425536 1.6902047075695310063 0.788430271281620354;-2.2520234103348077959 -1.3623610760019417842 0.58643590606324347281;2.0833402817557602482 2.0534094552649837517 1.2104702375837008699;-1.7444828269537828724 -1.6617802010127522561 1.7432004242243577252;-0.044471919073745452833 0.69704528622090422552 3.0135201404420293159;0.88299472438513026962 -2.2132473512261756632 -2.007536436869085783];

% Layer 2
b2 = 0.89888155747189890654;
LW2_1 = [-2.1905641224330785199 -0.20610529830049845179 0.96030067519542017251 0.30346710061981491124 0.01408937093181836242 0.68153323798417908552 0.98246257162682903985 0.41722741066732960125 -0.24032394685674607349 1.395770921504036588];

% Output 1
y1_step1.ymin = -1;
y1_step1.gain = 0.250728512194969;
y1_step1.xoffset = -3.99369680999518;

% ===== SIMULATION ========

x1=log10(Ed);

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
PAR_b = 10.^y1';

mat=[  0.1         -32.152       8.6118
    0.2122      -34.554       7.4676
    0.23809      -34.058       8.7968
    0.26715      -33.501       10.288
    0.29974      -32.876       11.962
    0.33632      -32.992       11.611
    0.37735      -33.122       11.217
    0.4234      -33.268       10.775
    0.47506      -29.976       5.7866
    0.53303      -30.016       5.2513
    0.59807       -30.06       4.6508
    0.67104       -30.57       4.6172
    0.75292       -29.15        4.577
    0.84479      -27.557       4.5319
    0.94787      -26.759       5.4663
    1.0635      -25.765       5.9184
    1.1933      -25.779       5.4639
    1.3389      -22.798       3.3904
    1.5023      -21.909       6.3067
    1.6856      -22.406       4.9738
    1.8913      -21.908       6.0759
    2.122      -18.468       5.0728
    2.3809      -16.717       6.5739
    2.6715      -15.752       4.9724
    2.9974      -14.896       6.6056
    3.3632      -13.207       3.9569
    3.7735      -11.269       4.2232
    4.234      -9.6052       6.6315
    4.7506      -7.4387       7.0555
    5.3303      -6.3571       8.2596
    5.9807      -3.9604       7.6006
    6.7104      -2.1452       5.5418
    7.5292     -0.64392       6.9823
    8.4479      0.82185       7.1562
    9.4787       2.1708       7.4852
    10.635       3.0682       8.3793
    11.933       4.0633       7.3601
    13.389       4.6963       8.0323
    15.023       5.0677       8.8866
    16.856       5.2124       8.5065
    18.913       5.0627       8.8228
    21.22       4.5972       10.346
    23.809       4.0695       11.007
    26.715       3.2747       10.404
    29.974       2.2568        12.46
    33.632       1.3623       9.9301
    37.735      0.27763       10.123
    42.34     -0.82735       10.067
    47.506      -1.7872       10.874
    53.303      -1.8461       12.408
    59.807      -1.4809       11.573
    67.104     -0.38744       11.344
    75.292      0.67433       12.599
    84.479       2.4153       13.152
    94.787       4.2855       14.077
    106.35         4.87       14.265
    119.33       5.1701       13.723
    133.89       4.8857       12.036
    150.23       5.6487       9.8324
    168.56       5.3319       9.4722
    189.13       3.1445       8.9049
    201             0            0
   1000             0            0];

ep50=interp1(log10(mat(:,1)),mat(:,2),log10(z));
IQR_ep=interp1(log10(mat(:,1)),mat(:,3),log10(z));

PAR=PAR_b./(1+ep50/100);
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
