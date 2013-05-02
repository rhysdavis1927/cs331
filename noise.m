(* ::Package:: *)

(*Mathematica Code for MLP*)
(*Normal random variables with variance 1*)
xrv:=2{Random[],Random[]}-1;
gauss:=Module[{t},t=xrv;t=If[t.t>=1,gauss,t[[1]]Sqrt[-2Log[t.t]/t.t]]];
noise1=Table[.5{gauss,gauss},{i,1000}]; (*Normal noise var=0.25*)
noise2=Table[.5{gauss,gauss},{i,1000}]; (*Normal noise var=0.25*)

trainSize=64;
hiddenNeurons=2;
(*Logistic function:*)
f[x_]=1/(1+Exp[-x]); (*works on vectors component-wise*)
(*For the XOR problem with n hidden neurons*)
(*Defining weight matrices in-hidden:*)
intoh[n_]:=Table[4.Random[]-2,{n},{3}]
(*and hidden-op:*)
htout[n_]:=Table[4.Random[]-2,{n+1}] (*extra one for offset*)
(*For the 2-hidden unit network:*)
(*First set up weight matrices:*)
weight1=intoh[hiddenNeurons]; (*for 2 hidden units*)
weight2=htout[hiddenNeurons];
(*Then a table for the errors:*)
etab=Table[0,{1000}];
etab2=Table[0,{100}];
(*and an offset vector:*)
ones=Table[-1,{64}];
train16 = {{ 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},{0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1}};
train16out=Abs[{1,-1}.train16];
train16wnoise=Join[train16+Transpose[Take[noise1,16]],{Take[ones,16]}];

train32 = {{ 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},{0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1}};
train32out=Abs[{1,-1}.train32];
train32wnoise=Join[train32+Transpose[Take[noise1,32]],{Take[ones,32]}];

train64 = {{ 0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},{0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1}};
train64out=Abs[{1,-1}.train64];
train64wnoise=Join[train64+Transpose[Take[noise1,64]],{Take[ones,64]}];
(*Target output*)
If[trainSize==16,trainop=train16out;train=train16wnoise;,0;]
If[trainSize==32,trainop=train32out;train=train32wnoise;,0;]
If[trainSize==64,trainop=train64out;train=train64wnoise;,0;]
(*{0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0};*)

intoh[n_]:=Table[4.Random[]-2,{n},{3}]
(*and hidden-op:*)
htout[n_]:=Table[4.Random[]-2,{n+1}] (*extra one for offset*)
(*For the 2-hidden unit network:*)
(*First set up weight matrices:*)
weight1=intoh[8] (*for 2 hidden units*)
weight2=htout[8]


train16wnoise
train16out
train32wnoise
train32out
train64wnoise
train64out





