clc, clear, close all

N = 1000;
A = 7;
B = 3;
f1 = 10;
f2 = 12;

ulaz = linspace(0,1.8,N);
f = A*sin(2*pi*f1*ulaz) + B*sin(2*pi*f2*ulaz);

izlaz = f + randn(1,N)*0.2*min(A,B);

figure("Name", "Funkcija(y(x)) i funkcija sa sumom(h(x))")
hold all
plot(ulaz, f, 'r')
plot(ulaz, izlaz, 'b')

ind = randperm(N);

ulazTrening = ulaz(:, ind(1:0.9*N));
izlazTrening = izlaz(:,ind(1:0.9*N));
ulazTest = ulaz(:, ind(0.9*N+1:N));
izlazTest = izlaz(:,ind(0.9*N+1:N));

net = fitnet([5 10 10 15 20]);
net.performFcn= 'mse';
net.divideFcn='';
net.trainParam.epochs= 350;
net.trainParam.goal= 0.000001;

[net, tr] = train(net,ulazTrening,izlazTrening);
figure
plotperform(tr)
figure
plotregression(izlazTrening, net(ulazTrening), "Regresija")


predTest = sim(net, ulazTest); 
figure, hold all
plot(ulazTest, izlazTest, 'o');
plot(ulazTest, predTest, '*');


fPred = sim(net,ulaz);
figure("Name", "Funkcija sa sumom i predikcija iste"), hold all
plot(ulaz, izlaz, 'b')
plot(ulaz, fPred, 'r', 'LineWidth', 3);
