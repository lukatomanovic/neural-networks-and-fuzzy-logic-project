clc, clear, close all

%% Ucitavanje podataka
podaci = csvread("CO2/CO2_dataset.csv",1,0);

%izlaz=podaci(:,8)';%TEY
%uzlaz=[podaci(:,1:7),podaci(:,9:11)]';

izlaz=podaci(:,10)';%CO
uzlaz=podaci(:,1:9)';

%% Priprema podataka

K1in=uzlaz(:,izlaz>0 & izlaz<9);
K1out=izlaz(izlaz>0 & izlaz<9);

K2in=uzlaz(:,izlaz>=9 & izlaz<18);
K2out=izlaz(izlaz>=9 & izlaz<18);

K3in=uzlaz(:,izlaz>=18 & izlaz<27);
K3out=izlaz(izlaz>=18 & izlaz<27);

K4in=uzlaz(:,izlaz>=27 & izlaz<36);
K4out=izlaz(izlaz>=27 & izlaz<36);

K5in=uzlaz(:,izlaz>=36 & izlaz<45);
K5out=izlaz(izlaz>=36 & izlaz<45);


brojKlasa = 5; %nbins
h = histogram(izlaz,brojKlasa);%prikaz da imamo podatke samo kad je mala zagadjenost

N1=length(K1out);
N2=length(K2out);
N3=length(K3out);
N4=length(K4out);
N5=length(K5out);

%K1 izdvajanje podataka
K1trening_in = K1in(:, 1:ceil(0.8*N1));
K1validacija_in=K1in(:,ceil(0.8*N1)+1:ceil(0.9*N1));
K1test_in=K1in(:,ceil(0.9*N1)+1:N1);

K1trening_out = K1out(1:ceil(0.8*N1));
K1validacija_out=K1out(ceil(0.8*N1)+1:ceil(0.9*N1));
K1test_out=K1out(ceil(0.9*N1)+1:N1);


%K2 izdvajanje podataka
K2trening_in = K2in(:, 1:ceil(0.8*N2));
K2validacija_in=K2in(:,ceil(0.8*N2)+1:ceil(0.9*N2));
K2test_in=K2in(:,ceil(0.9*N2)+1:N2);

K2trening_out = K2out(1:ceil(0.8*N2));
K2validacija_out=K2out(ceil(0.8*N2)+1:ceil(0.9*N2));
K2test_out=K2out(ceil(0.9*N2)+1:N2);


%K3 izdvajanje podataka
K3trening_in = K3in(:, 1:ceil(0.8*N3));
K3validacija_in=K3in(:,ceil(0.8*N3)+1:ceil(0.9*N3));
K3test_in=K3in(:,ceil(0.9*N3)+1:N3);

K3trening_out = K3out(1:ceil(0.8*N3));
K3validacija_out=K3out(ceil(0.8*N3)+1:ceil(0.9*N3));
K3test_out=K3out(ceil(0.9*N3)+1:N3);

%K4 izdvajanje podataka
K4trening_in = K4in(:, 1:ceil(0.8*N4));
K4validacija_in=K4in(:,ceil(0.8*N4)+1:ceil(0.9*N4));
K4test_in=K4in(:,ceil(0.9*N4)+1:N4);

K4trening_out = K4out(1:ceil(0.8*N4));
K4validacija_out=K4out(ceil(0.8*N4)+1:ceil(0.9*N4));
K4test_out=K4out(ceil(0.9*N4)+1:N4);


%K5 izdvajanje podataka
K5trening_in = K5in(:, 1:ceil(0.8*N5));
K5validacija_in=K5in(:,ceil(0.8*N5)+1:ceil(0.9*N5));
K5test_in=K5in(:,ceil(0.9*N5)+1:N5);

K5trening_out = K5out(1:ceil(0.8*N5));
K5validacija_out=K5out(ceil(0.8*N5)+1:ceil(0.9*N5));
K5test_out=K5out(ceil(0.9*N5)+1:N5);


%grupisanje u skupove podataka za treniranje, validaciju i test
ulazTrening=[K1trening_in,K2trening_in,K3trening_in,K4trening_in,K5trening_in];
izlazTrening=[K1trening_out,K2trening_out,K3trening_out,K4trening_out,K5trening_out];

ulazVal=[K1validacija_in,K2validacija_in,K3validacija_in,K4validacija_in,K5validacija_in];
izlazVal=[K1validacija_out,K2validacija_out,K3validacija_out,K4validacija_out,K5validacija_out];

ulazTest=[K1test_in,K2test_in,K3test_in,K4test_in,K5test_in];
izlazTest=[K1test_out,K2test_out,K3test_out,K4test_out,K5test_out];

%promesamo podatke
indTrening=randperm(length(izlazTrening));
ulazTrening=ulazTrening(:,indTrening);
izlazTrening=izlazTrening(indTrening);

indVal=randperm(length(izlazVal));
ulazVal=ulazVal(:,indVal);
izlazVal=izlazVal(indVal);

indTest=randperm(length(izlazTest));
ulazTest=ulazTest(:,indTest);
izlazTest=izlazTest(indTest);

ulazSve=[ulazTrening,ulazVal];
izlazSve=[izlazTrening,izlazVal];
N=length(izlazSve);
Ntrening=length(izlazTrening);
Nval=length(izlazVal);


%samo test

% structure =[10 25 10];
%
% net = fitnet(structure);
% net.trainFcn='trainbr';
% net.performFcn='mse';
% net.layers{1}.transferFcn ='logsig';
% net.layers{2}.transferFcn ='poslin';
% net.layers{3}.transferFcn ='tansig';
%
% net.divideFcn = 'divideind';
% net.divideParam.trainInd = 1:Ntrening;
% net.divideParam.testInd = [];
% net.divideParam.valInd = Ntrening+1:Ntrening+Nval;
% net.trainParam.max_fail = 3;
% net.trainParam.goal = 10e-4;
% net.trainParam.min_grad = 10e-8;
% net.trainParam.epochs = 1000;
%
%
%
%
% [net, tr] = train(net, ulazSve, izlazSve);
%
% figure
% plotperform(tr)
% figure
% plotregression(izlazTrening, net(ulazTrening), "Regresija")
%
%
% predTest = sim(net, ulazTest);
% figure, hold all
% plot(ulazTest, izlazTest, 'bo');
% plot(ulazTest, predTest, 'r.');
%
%
% predVal = net(ulazVal);
% errors = gsubtract(izlazVal,predVal);
%
% sum=0;
% for i =1:Nval
%     sum=sum+errors(i)*errors(i);
% end
% sum=sum/Nval;
%
% mse_error=sqrt(sum);



%% Kreiranje i obucavanje NM
bestF1=0;
min_mse_error=50;
iter=0;
for structure = {[10 50 10],[20 40 30],[10 20 10]}
    for weight=[1 2 3]
        for f1 = {'tansig', 'logsig','poslin'}
            for f2 = {'tansig', 'logsig','poslin'}
                for f3 = {'tansig', 'logsig','poslin'}
                    for regularization = 0:0.1:1
                        disp(iter);
                        iter=iter+1;
                        net = fitnet(structure{1});
                        %net.trainFcn='trainbr';
                        net.performFcn='mse';
                        
                        %for i = 1:length(structure{1})
                        %    net.layers{i}.transferFcn = f{1};
                        %end
                        net.layers{1}.transferFcn = f1{1};
                        net.layers{2}.transferFcn = f2{1};
                        net.layers{3}.transferFcn = f3{1};
                        
                        net.divideFcn = 'divideind';
                        net.divideParam.trainInd = 1:Ntrening;
                        net.divideParam.testInd = [];
                        net.divideParam.valInd = Ntrening+1:Ntrening+Nval;
                        net.trainParam.max_fail = 10;
                        net.trainParam.goal = 10e-8;
                        net.trainParam.min_grad = 10e-8;
                        net.trainParam.epochs = 1000;
                        
                        
                        net.performParam.regularization = regularization;
%                         net.divideFcn = 'divideind';
%                         net.divideParam.trainInd = 1:Ntrening;
%                         net.divideParam.testInd = [];
%                         net.divideParam.valInd = Ntrening+1:Ntrening+Nval;
                        
                        net.trainParam.showWindow=false;
                        net.trainParam.showCommandLine=true;
                        
                        
                        W= ones(1,length(izlazSve));
                        W(izlazSve>10 & izlazSve<15)=weight;
                        W(izlazSve>=15)=2*weight;
                        
                        [net, tr] = train(net, ulazSve, izlazSve,[],[],W);
                        
                        %                 figure
                        %                 plotperform(tr)
                        %                 figure
                        %                 plotregression(izlazTrening, net(ulazTrening), "Regresija na treningu")
                        disp(net.performParam.regularization);
                        
                        
                        predVal = net(ulazVal);
                        errors = gsubtract(izlazVal,predVal);
                        
                        sum=0;
                        for i =1:Nval
                            sum=sum+errors(i)*errors(i);
                        end
                        sum=sum/Nval;
                        
                        mse_error=sqrt(sum);
                        
                        if mse_error<min_mse_error
                            min_mse_error=mse_error;
                            best_weight=weight;
                            best_reg=regularization;
                            best_f1=f1{1};
                            best_f2=f2{1};
                            best_f3=f3{1};
                            best_epoch=tr.best_epoch;
                            best_iter=iter;
                            best_structure=structure{1};
                            best_goal=tr.goal;
                        end
                    end
                end
            end
        end
    end
end

%% Obuka mreze sa najboljim hiperparametrima
net = fitnet(best_structure);

net.performFcn='mse';

%for i = 1:length(structure{1})
%    net.layers{i}.transferFcn = f{1};
%end
net.layers{1}.transferFcn = best_f1;
net.layers{2}.transferFcn = best_f2;
net.layers{3}.transferFcn = best_f3;

net.divideFcn = '';

% net.trainParam.max_fail = 10;
% net.trainParam.goal = 10e-4;
% net.trainParam.min_grad = 10e-8;
net.trainParam.epochs = 10;
net.trainParam.max_fail = 10;


net.performParam.regularization = best_reg;
net.divideFcn = '';
%net.divideParam.trainInd = 1:Ntrening;
%net.divideParam.testInd = [];
%net.divideParam.valInd = Ntrening+1:Ntrening+Nval;

%net.trainParam.showWindow=false;
%net.trainParam.showCommandLine=true;


W= ones(1,length(izlazSve));
W(izlazSve>10 & izlazSve<15)=best_weight;
W(izlazSve>=15)=2*best_weight;

[net, tr] = train(net, ulazSve, izlazSve,[],[],W);

predSve = net(ulazSve);

figure
plotregression(izlazSve, predSve, "Regresija finalno")

errors = gsubtract(izlazSve,predSve);


sum=0;
for i =1:Nval
    sum=sum+errors(i)*errors(i);
end
sum=sum/Nval;

mse_error=sqrt(sum);

%% Testiranje NM

net = fitnet(best_structure);

net.performFcn='mse';

%for i = 1:length(structure{1})
%    net.layers{i}.transferFcn = f{1};
%end
net.layers{1}.transferFcn = best_f1;
net.layers{2}.transferFcn = best_f2;
net.layers{3}.transferFcn = best_f3;

net.divideFcn = '';

% net.trainParam.max_fail = 10;
% net.trainParam.goal = 10e-4;
% net.trainParam.min_grad = 10e-8;
net.trainParam.epochs = 10;
net.trainParam.max_fail = 10;


net.performParam.regularization = best_reg;
net.divideFcn = '';
%net.divideParam.trainInd = 1:Ntrening;
%net.divideParam.testInd = [];
%net.divideParam.valInd = Ntrening+1:Ntrening+Nval;

%net.trainParam.showWindow=false;
%net.trainParam.showCommandLine=true;


W= ones(1,length(ulazTest));
W(izlazTest>10 & izlazTest<15)=best_weight;
W(izlazTest>=15)=2*best_weight;

[net, tr] = train(net, ulazTest, izlazTest,[],[],W);

predTest = net(ulazTest);

figure
plotregression(izlazTest, predTest, "Regresija - test skup")

errors = gsubtract(izlazTest,predTest);


sum=0;
for i =1:Nval
    sum=sum+errors(i)*errors(i);
end
sum=sum/Nval;

mse_error=sqrt(sum);