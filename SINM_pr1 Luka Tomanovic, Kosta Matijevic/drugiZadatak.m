clc, clear, close all

%% Ucitavanje podataka
load('dataset1.mat');
data = pod';

ob = data(1:2, :);
klasa = data(3, :);
N = length(klasa);

K1 = ob(:, klasa == 1);
K2 = ob(:, klasa == 2);
K3 = ob(:, klasa == 3);

figure, hold all
plot(K1(1, :), K1(2, :), 'o')
plot(K2(1, :), K2(2, :), '*')
plot(K3(1, :), K3(2, :), 'd')

%% One-hot encoding
izlaz = zeros(3, N);

izlaz(1, klasa == 1) = 1;
izlaz(2, klasa == 2) = 1;
izlaz(3, klasa == 3) = 1;

ulaz = ob;

%% Podela na trening i test skup
ind = randperm(N);
indTrening = ind(1 : 0.9*N);
indTest = ind(0.9*N+1 : N);

ulazTrening = ulaz(:, indTrening);
izlazTrening = izlaz(:, indTrening);

ulazTest = ulaz(:, indTest);
izlazTest = izlaz(:, indTest);

%% Kreiranje NM1 - optimalna
arhitektura = [20 15 10];
net = patternnet(arhitektura);

for i = 1 : length(arhitektura)
    net.layers{i}.transferFcn = 'poslin';
end
net.layers{length(arhitektura) + 1}.transferFcn = 'softmax';

%net.performFcn = 'crossentropy';
%net.performFcn = 'mse';
% net.performParam.regularization = 0.2;

net.divideFcn = '';
%net.divideFcn = 'dividerand';
%net.divideParam.trainRatio = 0.9;
%net.divideParam.valRatio = 0.1;
%net.divideParam.testRatio = 0;

net.trainParam.epochs = 2000;
net.trainParam.goal = 1e-4;
net.trainParam.min_grad = 1e-5;
net.trainParam.max_fail = 20;

%% Kreiranje NM2 - underfitting
arhitektura2 = [2 1];
net2 = patternnet(arhitektura2);

for i = 1 : length(arhitektura2)
    net2.layers{i}.transferFcn = 'poslin';
end
net2.layers{length(arhitektura2) + 1}.transferFcn = 'softmax';

%net.performFcn = 'crossentropy';
%net.performFcn = 'mse';
% net.performParam.regularization = 0.2;

net2.divideFcn = '';
%net.divideFcn = 'dividerand';
%net.divideParam.trainRatio = 0.9;
%net.divideParam.valRatio = 0.1;
%net.divideParam.testRatio = 0;

net2.trainParam.epochs = 2000;
net2.trainParam.goal = 1e-4;
net2.trainParam.min_grad = 1e-5;
net2.trainParam.max_fail = 20;

%% Kreiranje NM3 - overfitting
arhitektura3 = [500 100 100 200 200 500 1];
net3 = patternnet(arhitektura3);

for i = 1 : length(arhitektura3)
    net3.layers{i}.transferFcn = 'poslin';
end
net3.layers{length(arhitektura3) + 1}.transferFcn = 'softmax';

%net.performFcn = 'crossentropy';
%net.performFcn = 'mse';
% net.performParam.regularization = 0.2;

net3.divideFcn = '';
%net.divideFcn = 'dividerand';
%net.divideParam.trainRatio = 0.9;
%net.divideParam.valRatio = 0.1;
%net.divideParam.testRatio = 0;

net3.trainParam.epochs = 2000;
net3.trainParam.goal = 1e-4;
net3.trainParam.min_grad = 1e-5;
net3.trainParam.max_fail = 20;
%% Treniranje NM
[net, tr] = train(net, ulazTrening, izlazTrening);

figure
plotperform(tr);
title("Performance NM1 OPTIMAL");

predTrening = net(ulazTrening);
figure, plotconfusion(izlazTrening, predTrening)
title("Confusion Matrix NM1- trening");

%% Treniranje NM2
[net2, tr2] = train(net2, ulazTrening, izlazTrening);

figure
plotperform(tr2);
title("Performanse NM2 UNDERFITTING");

predTrening = net2(ulazTrening);
figure, plotconfusion(izlazTrening, predTrening)
title("Confusion Matrix NM2- trening");

%% Treniranje NM3
[net3, tr3] = train(net3, ulazTrening, izlazTrening);

figure
plotperform(tr3);
title("Performanse NM3 OVERFITTING");

predTrening = net3(ulazTrening);
figure, plotconfusion(izlazTrening, predTrening)
title("Confusion Matrix NM3- trening");

%% Performanse NM
predTest = net(ulazTest);
figure, plotconfusion(izlazTest, predTest)
title("Confusion Matrix NM1 - test");

%za prikaz greske na validacionom skupu ako on deli podatke
%predVal = net(ulazTrening(:, tr.valInd));
%figure, plotconfusion(izlazTrening(:, tr.valInd), predVal)


[c, cm] = confusion(izlazTest, predTest);
cm = cm';

% K1 kao klasa od interesa
P = cm(1, 1)/sum(cm(1, :));%ONO STO JE POGODJENO ZA KLASU 1 PODELJENO SA SVIM ONIM STO JE MREZA PROGLASILA KAO KLASU 1
R = cm(1, 1)/sum(cm(:, 1)); %ONO STO JE POGODJENO OD KLASE 1
F1 = 2*P*R/(P+R);

%% Performanse NM2
predTest2 = net2(ulazTest);
figure, plotconfusion(izlazTest, predTest2)
title("Confusion Matrix NM2 UNDERFITTING - test");

[c, cm] = confusion(izlazTest, predTest2);
cm = cm';

% K1 kao klasa od interesa
P2 = cm(1, 1)/sum(cm(1, :));%ONO STO JE POGODJENO ZA KLASU 1 PODELJENO SA SVIM ONIM STO JE MREZA PROGLASILA KAO KLASU 1
R2 = cm(1, 1)/sum(cm(:, 1)); %ONO STO JE POGODJENO OD KLASE 1
F12 = 2*P2*R2/(P2+R2);

%% Performanse NM3
predTest3 = net3(ulazTest);
figure, plotconfusion(izlazTest, predTest3)
title("Confusion Matrix NM3 OVERFITTING - test");

[c, cm] = confusion(izlazTest, predTest3);
cm = cm';

% K1 kao klasa od interesa
P3 = cm(1, 1)/sum(cm(1, :));%ONO STO JE POGODJENO ZA KLASU 1 PODELJENO SA SVIM ONIM STO JE MREZA PROGLASILA KAO KLASU 1
R3 = cm(1, 1)/sum(cm(:, 1)); %ONO STO JE POGODJENO OD KLASE 1
F13 = 2*P3*R3/(P3+R3);
%% Granica odlucivanja NM1
Ntest = 500;
x1 = repmat(linspace(-4, 4, Ntest), 1, Ntest);
x2 = repelem(linspace(-4, 4, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net(ulazGO);
[vr, klasaGO] = max(predGO);%izvlaci maximum po kolonama i indekse vrste gde se nalaze

K1go = ulazGO(:, predGO(1, :) >= 0.7);
K2go = ulazGO(:, predGO(2, :) >= 0.7);
K3go = ulazGO(:, predGO(3, :) >= 0.7);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')

%% Granica odlucivanja NM2
Ntest = 500;
x1 = repmat(linspace(-4, 4, Ntest), 1, Ntest);
x2 = repelem(linspace(-4, 4, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net2(ulazGO);
[vr, klasaGO] = max(predGO);%izvlaci maximum po kolonama i indekse vrste gde se nalaze

K1go = ulazGO(:, predGO(1, :) >= 0.3);
K2go = ulazGO(:, predGO(2, :) >= 0.3);
K3go = ulazGO(:, predGO(3, :) >= 0.3);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')

%% Granica odlucivanja NM3
Ntest = 500;
x1 = repmat(linspace(-4, 4, Ntest), 1, Ntest);
x2 = repelem(linspace(-4, 4, Ntest), Ntest);
ulazGO = [x1; x2];

predGO = net3(ulazGO);
[vr, klasaGO] = max(predGO);%izvlaci maximum po kolonama i indekse vrste gde se nalaze

K1go = ulazGO(:, predGO(1, :) >= 0.3);
K2go = ulazGO(:, predGO(2, :) >= 0.3);
K3go = ulazGO(:, predGO(3, :) >= 0.3);

figure, hold all
plot(K1go(1, :), K1go(2, :), '.')
plot(K2go(1, :), K2go(2, :), '.')
plot(K3go(1, :), K3go(2, :), '.')
plot(K1(1, :), K1(2, :), 'bo')
plot(K2(1, :), K2(2, :), 'r*')
plot(K3(1, :), K3(2, :), 'yd')