%.,mnb  nm%% Gustav Sto. Tomas, A15358078, COGS109 Homework 1

%%% 4.a

clc;
clear;
close all;

Income = csvread('Income2.csv'); %read csv file as input
Income; 

Income = Income(2:end,:); %remove first row of strings (in Octave, this row is displayed as 0.00000)

x = Income(:,2); % x axis = second column (education)
y = Income(:,4); % y axis = fourth column (income)

figure 1 %create figure for plot
scatter(x,y) %scatterplot x and y
ylabel('Income in k $')
xlabel('Level of education')
title('Correlation of education and income level')

%%% 4.b
mean = mean(y);

%%% 4.c
std = std(y);

%%% 4.d
SEM = std/(sqrt(length(y)));

%%% 4.e

%HigherEd = zeros(30,4);


x_categorical = (x>=16);
%for i=1:1:(size(x_categorical,1))
%  if i == 1
%  HigherEd = [i,0]
%  end
%  end

HigherEd = [y,x_categorical];
%for x_categorical=1:1:size(HigherEd)

%x_categorical_0 = (x<16);

%HigherEd = [x_categorical, x_categorical_0];
%pkg load statistics

%figure 2
%for x_categorical=1:1:size(x_categorical)

%plot(x,y)
%ylabel('Income in $$$')
%xlabel('HigherEd')
%hold on
%end
%end
%end
%end

figure 2
boxplot(HigherEd);
ylabel('Income level')
xlabel('Higher education Y/N')
title('Correlation of income level and higher education')
%end