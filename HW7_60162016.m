%%60162016 김아람
%%(a)
clear; close all; clc;
x= 1:0.1:10;
plot(x, cos(x));
xlabel("x");
ylabel("cos(x)");
title("Simple 2-D plot");

%%(b)
data = csvread("hw7data.txt") 

%%(c)
x = data(:,[1])
y = data(:,[2])
m=length(y)

%%(d)
figure;
plot(x,y,'rx')
pause;

%%(e)- cost function 
X = [ones(m,1), data(:,1)];
theta = zeros(2,1);
%% define computeCost function
function cost = computeCost(X,y,theta)
  m = length(y);
  cost = 0;
  predictions = X*theta;  
  sqrErrors   = (predictions - y).^2; 
  cost = 1/(2*m) * sum(sqrErrors);
  
endfunction
computeCost(X,y,theta)
fprintf("코드진행을 위해서 enter를 입력해주세요.")
pause;

%%(f)
iterations = 1500;
alpha = 0.01;

function [theta, history] = gradientDescent(X, y, theta, alpha, num_iters)
  m = length(y);
  history = zeros(num_iters, 1);
  for iter = 1:num_iters
    x = X(:,2);
    h = theta(1) + (theta(2)*x);

    theta_zero = theta(1) - alpha * (1/m) * sum(h-y);
    theta_one  = theta(2) - alpha * (1/m) * sum((h - y).* x);

    theta = [theta_zero; theta_one];
    history(iter) = computeCost(X, y, theta);
  endfor
  disp(min(history));
endfunction

fprintf("실행중.....")
[theta,history]= gradientDescent(X,y,theta,alpha,iterations);

%이전에 그린 데이터에 학습한 결과 그리기
hold on;
plot(X(:,2), X*theta, '-')
legend('Training data' , 'Linear regression')
hold off;

%%(g)
predict1 = [1,20]*theta;
predict2 = [1, 5.5]*theta;

fprintf('For x=20, y= %f\n', predict1);
fprintf('For x=x5.5, y= %f\n', predict2);
fprintf("코드 진행을 위해서 enter를 입력해주세요.\n")
pause;

%%전체학습 과정에서의 cost변화 plotting 하기
figure;
plot([1:iterations],history)
xlabel('num of iterations')
ylabel('Cost')
fprintf("코드진행을 위해서 enter를 입력해주세요.\n")
pause;