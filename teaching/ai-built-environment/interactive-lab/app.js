"use strict";

const MODULE_GROUPS = [
  {name:"Foundations",items:[
    {id:"linear",title:"Linear Regression",status:"live",summary:"Fit a line to editable data and connect residuals, loss and gradient descent.",visual:"A prediction line, residuals and a live loss surface.",interaction:"Move data points and parameters, then take individual gradient-descent steps."},
    {id:"gradient",title:"Gradient Descent",status:"live",summary:"Explore how optimizers move through a loss landscape.",visual:"Contours, gradients and competing optimizer paths.",interaction:"Change learning rate, momentum, starting point and surface shape."},
    {id:"backprop",title:"Backpropagation",status:"live",summary:"Trace a prediction forward, then send error gradients backward through a computation graph.",visual:"A live computation graph with local derivatives and accumulated gradients.",interaction:"Change inputs and weights, step through forward and backward passes, then apply one update."},
    {id:"training",title:"Neural Network Playground",status:"live",summary:"Build and train a small network while watching its internal representations and decision boundary evolve.",visual:"Feature maps, neuron activations, signed weighted connections, output boundary and loss curves.",interaction:"Play, pause or step through training; change features and architecture; click any neuron; then compare frozen models."},
    {id:"pca",title:"PCA & Dimensionality Reduction",status:"live",summary:"Compress data while distinguishing variance-preserving projections from neighborhood embeddings.",visual:"Editable principal axes, projected points, variance bars and a live t-SNE neighborhood map.",interaction:"Move data, rotate projection axes, change perplexity and compare different embedding seeds."},
    {id:"generalization",title:"Generalization",status:"live",summary:"Understand training, validation, testing, underfitting and overfitting.",visual:"Training and test errors changing with model complexity.",interaction:"Change noise, dataset size, split and model flexibility."}
  ]},
  {name:"Classical ML",items:[
    {id:"logistic",title:"Logistic Regression",status:"live",summary:"Turn a linear score into a probability and a decision boundary.",visual:"A sigmoid, probability heatmap, boundary and confusion matrix.",interaction:"Move points and adjust weights, threshold and class balance."},
    {id:"knn",title:"K-Nearest Neighbours",status:"live",summary:"Classify a point using nearby examples rather than a fitted equation.",visual:"Neighbourhood circles, votes and a decision map.",interaction:"Move the query point and change K, distance and feature scaling."},
    {id:"tree",title:"Decision Trees",status:"live",summary:"Build predictions through a sequence of data splits.",visual:"Linked decision regions, candidate splits and the growing tree.",interaction:"Advance one split at a time and change depth or impurity measure."},
    {id:"ensembles",title:"Ensembles & Boosting",status:"live",summary:"Combine weak models through bagging, forests and boosting.",visual:"Bootstrap samples, learner votes and residual fitting rounds.",interaction:"Change learner count, sample size, feature choice and learning rate."},
    {id:"svm",title:"Support Vector Machines",status:"live",summary:"Find a wide separating margin and introduce kernels.",visual:"Margins, support vectors and linear or curved boundaries.",interaction:"Move points and adjust regularization and kernel parameters."},
    {id:"kmeans",title:"K-Means Clustering",status:"live",summary:"Discover groups by alternating assignment and centroid updates.",visual:"Clusters, Voronoi regions and moving centroids.",interaction:"Place centroids, change K and step through the two update phases."},
    {id:"evaluation",title:"Evaluation & Thresholds",status:"live",summary:"Connect model outputs to errors, metrics and decisions.",visual:"Confusion matrix, ROC curve and precision–recall trade-offs.",interaction:"Change threshold, class balance and the cost of each error."},
    {id:"arena",title:"Model Comparison Arena",status:"live",summary:"See how different methods interpret exactly the same data.",visual:"Multiple decision boundaries shown side by side.",interaction:"Edit one dataset and compare logistic, KNN, tree and kernel methods."}
  ]},
  {name:"Deep Learning",items:[
    {id:"neural",title:"Neuron & MLP",status:"live",summary:"Build nonlinear functions by connecting simple computational units.",visual:"Network graph, weighted sums and per-neuron activations.",interaction:"Change inputs, weights, activation and network depth."},
    {id:"cnn",title:"CNN Playground",status:"live",summary:"Trace an editable image through convolution, activation and pooling while comparing several filters.",visual:"A highlighted receptive field, editable kernel arithmetic, feature maps and pooled outputs.",interaction:"Draw pixels, edit filter weights, scan locations, change geometry and compare a multi-filter feature stack."},
    {id:"rnn",title:"RNN & LSTM Playground",status:"live",summary:"Edit a sequence and compare recurrent state, gated memory and first-input influence through time.",visual:"An unrolled timeline, live LSTM gates, hidden-state traces and sensitivity curves.",interaction:"Edit or extend the sequence, inspect time steps and gates, then compare RNN and LSTM memory on identical inputs."},
    {id:"autoencoder",title:"Autoencoder & Latent Space Playground",status:"live",summary:"Compress editable images, compare reconstruction and denoising, then navigate a two-dimensional latent space.",visual:"Clean and noisy inputs, a live bottleneck, reconstruction trade-off curve and draggable latent map.",interaction:"Draw pixels, change code width and learning objective, or move through latent space to generate blended outputs."}
  ]},
  {name:"Generative & Foundation Models",items:[
    {id:"diffusion",title:"Diffusion Models",status:"live",summary:"Learn generation as gradual noising followed by iterative denoising.",visual:"Clean signal, noisy state, denoising estimate and noise schedule.",interaction:"Move through noise time, alter the schedule and compare denoising estimates."},
    {id:"transformer",title:"Attention & Transformers",status:"live",summary:"Use queries, keys and values to route information between tokens.",visual:"Attention connections, full attention matrix and live score arithmetic.",interaction:"Change the query token, attention head, masking and temperature."},
    {id:"llm",title:"Large Language Models",status:"live",summary:"Connect next-token probabilities to generated sequences.",visual:"Context window, token probabilities and an autoregressive generation path.",interaction:"Change temperature, top-k, context and decoding without an external model."},
    {id:"vlm",title:"Vision–Language Models",status:"live",summary:"Connect visual features and language in a shared representation.",visual:"Image patches, candidate descriptions, relevance and cross-modal similarity.",interaction:"Select or mask image patches and compare prepared descriptions."}
  ]}
];

const $ = selector => document.querySelector(selector);
const clamp = (value,min,max) => Math.max(min,Math.min(max,value));
const fmt = (value,digits=3) => Number.isFinite(value) ? value.toFixed(digits) : "—";

const navList = $("#navList");
MODULE_GROUPS.forEach(group => {
  const wrapper = document.createElement("section");
  wrapper.className = "nav-group";
  const heading = document.createElement("h2");
  heading.textContent = group.name;
  wrapper.appendChild(heading);
  group.items.forEach(item => {
    item.category = group.name;
    const button = document.createElement("button");
    button.type = "button";
    button.className = "module-button" + (item.id === "linear" ? " active" : "");
    button.dataset.module = item.id;
    button.innerHTML = `<span>${item.title}</span><small>Live</small>`;
    button.addEventListener("click",() => openModule(item));
    wrapper.appendChild(button);
  });
  navList.appendChild(wrapper);
});

function openModule(item){
  document.querySelectorAll(".module-button").forEach(button => button.classList.toggle("active",button.dataset.module === item.id));
  const isLinear = item.id === "linear";
  const isLogistic = item.id === "logistic";
  const conceptIds=["gradient","backprop","training","pca","generalization","knn","tree","ensembles","svm","kmeans","evaluation","arena","neural","cnn","rnn","autoencoder","diffusion","transformer","llm","vlm"];
  const isConcept=conceptIds.includes(item.id);
  $("#linearModule").classList.toggle("active",isLinear);
  $("#linearModule").hidden = !isLinear;
  $("#logisticModule").classList.toggle("active",isLogistic);
  $("#logisticModule").hidden = !isLogistic;
  $("#conceptModule").classList.toggle("active",isConcept);
  $("#conceptModule").hidden = !isConcept;
  stopAutoStep();
  logStopAutoStep();
  if(window.stopConceptAnimation)window.stopConceptAnimation();
  if(isLinear){
    requestAnimationFrame(render);
  }else if(isLogistic){
    requestAnimationFrame(logRender);
  }else if(window.activateConcept){
    requestAnimationFrame(()=>window.activateConcept(item.id));
  }
  if(window.updateLearningGuide)window.updateLearningGuide(item);
  if(window.innerWidth <= 820){
    $("#labNav").classList.remove("open");
    $("#navToggle").setAttribute("aria-expanded","false");
  }
}

$("#navToggle").addEventListener("click",() => {
  const open = $("#labNav").classList.toggle("open");
  $("#navToggle").setAttribute("aria-expanded",String(open));
});

function seededRandom(seed){
  let value = seed >>> 0;
  return () => {
    value = (1664525 * value + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function buildDataset(type){
  const random = seededRandom({clean:12,noisy:27,outlier:41,curve:63}[type] || 12);
  const points = [];
  for(let i=0;i<17;i++){
    const x = -4.2 + i * .52;
    let y;
    if(type === "curve") y = .34 * x * x - 2.1 + (random()-.5)*.5;
    else {
      const noise = type === "clean" ? .34 : .95;
      y = 1.05*x + .65 + (random()-.5)*2*noise;
    }
    points.push({x:clamp(x,-4.7,4.7),y:clamp(y,-4.7,4.7),split:i%5===0?"test":"train"});
  }
  if(type === "outlier"){
    points.splice(13,0,{x:3.6,y:-3.8,split:"train"});
  }
  return points;
}

const state = {
  points:buildDataset("clean"),w:0,b:0,alpha:.02,selected:2,dragging:-1,
  trajectory:[{w:0,b:0}],autoTimer:null,autoSteps:0,emphasis:"",
  lossView:"3d",lossYaw:.72,lossTilt:.43,lossDragging:false,lossPointer:{x:0,y:0}
};

const dataCanvas = $("#dataCanvas");
const lossCanvas = $("#lossCanvas");
const dataContext = dataCanvas.getContext("2d");
const lossContext = lossCanvas.getContext("2d");
const RANGE = {min:-5,max:5};
const PADDING = {left:46,right:18,top:18,bottom:38};

function fitCanvas(canvas,context){
  const ratio = Math.min(window.devicePixelRatio || 1,2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(280,Math.round(rect.width));
  const height = Math.max(240,Math.round(rect.height));
  if(canvas.width !== Math.round(width*ratio) || canvas.height !== Math.round(height*ratio)){
    canvas.width = Math.round(width*ratio); canvas.height = Math.round(height*ratio);
    context.setTransform(ratio,0,0,ratio,0,0);
  }
  return {width,height};
}

function plotArea(size){return {x:PADDING.left,y:PADDING.top,w:size.width-PADDING.left-PADDING.right,h:size.height-PADDING.top-PADDING.bottom}}
function xPixel(value,area){return area.x+(value-RANGE.min)/(RANGE.max-RANGE.min)*area.w}
function yPixel(value,area){return area.y+area.h-(value-RANGE.min)/(RANGE.max-RANGE.min)*area.h}
function xValue(pixel,area){return RANGE.min+(pixel-area.x)/area.w*(RANGE.max-RANGE.min)}
function yValue(pixel,area){return RANGE.min+(area.y+area.h-pixel)/area.h*(RANGE.max-RANGE.min)}

function prediction(x){return state.w*x+state.b}
function subset(split){return state.points.filter(point => point.split===split)}
function mse(points,w=state.w,b=state.b){
  if(!points.length) return NaN;
  return points.reduce((sum,point)=>sum+(point.y-(w*point.x+b))**2,0)/points.length;
}
function gradient(){
  const points=subset("train");
  if(!points.length)return {dw:0,db:0};
  let dw=0,db=0;
  points.forEach(point=>{const error=state.w*point.x+state.b-point.y;dw+=2*point.x*error;db+=2*error});
  return {dw:dw/points.length,db:db/points.length};
}

function drawGrid(context,area){
  context.save();context.strokeStyle="#dfe7ed";context.lineWidth=1;context.fillStyle="#657587";context.font="12px system-ui";
  for(let value=-4;value<=4;value+=2){
    const xp=xPixel(value,area),yp=yPixel(value,area);
    context.beginPath();context.moveTo(xp,area.y);context.lineTo(xp,area.y+area.h);context.stroke();
    context.beginPath();context.moveTo(area.x,yp);context.lineTo(area.x+area.w,yp);context.stroke();
    context.fillText(String(value),xp-5,area.y+area.h+19);context.fillText(String(value),area.x-26,yp+4);
  }
  context.strokeStyle="#92a3b0";context.lineWidth=1.3;
  context.beginPath();context.moveTo(xPixel(0,area),area.y);context.lineTo(xPixel(0,area),area.y+area.h);context.stroke();
  context.beginPath();context.moveTo(area.x,yPixel(0,area));context.lineTo(area.x+area.w,yPixel(0,area));context.stroke();
  context.fillStyle="#536477";context.font="600 12px system-ui";context.fillText("x",area.x+area.w-3,area.y+area.h+30);context.fillText("y",area.x-35,area.y+9);
  context.restore();
}

function drawDataPlot(){
  const size=fitCanvas(dataCanvas,dataContext),area=plotArea(size),context=dataContext;
  context.clearRect(0,0,size.width,size.height);drawGrid(context,area);
  context.save();context.beginPath();context.rect(area.x,area.y,area.w,area.h);context.clip();
  if($("#showResiduals").checked){
    context.lineWidth=1.4;context.strokeStyle="#e4a066aa";
    state.points.forEach(point=>{
      if(point.split==="test"&&!$("#showTest").checked)return;
      context.beginPath();context.moveTo(xPixel(point.x,area),yPixel(point.y,area));context.lineTo(xPixel(point.x,area),yPixel(prediction(point.x),area));context.stroke();
    });
  }
  context.strokeStyle=state.emphasis==="slope"||state.emphasis==="intercept"?"#d86f1f":"#c84d4d";context.lineWidth=state.emphasis?4:3;
  context.beginPath();context.moveTo(xPixel(RANGE.min,area),yPixel(prediction(RANGE.min),area));context.lineTo(xPixel(RANGE.max,area),yPixel(prediction(RANGE.max),area));context.stroke();
  state.points.forEach((point,index)=>{
    if(point.split==="test"&&!$("#showTest").checked)return;
    const x=xPixel(point.x,area),y=yPixel(point.y,area),selected=index===state.selected;
    context.beginPath();context.arc(x,y,selected?7:5.5,0,Math.PI*2);
    context.fillStyle=point.split==="test"?"#fff":"#176ca4";context.fill();context.lineWidth=point.split==="test"?2.5:1.3;context.strokeStyle=point.split==="test"?"#dc7728":"#0f527f";context.stroke();
    if(selected||state.emphasis==="point"){
      context.beginPath();context.arc(x,y,10,0,Math.PI*2);context.strokeStyle="#f0ad4e";context.lineWidth=2;context.stroke();
    }
    if(state.emphasis==="residual"&&selected){
      context.beginPath();context.moveTo(x,y);context.lineTo(x,yPixel(prediction(point.x),area));context.strokeStyle="#d86f1f";context.lineWidth=4;context.stroke();
    }
  });
  context.restore();
}

function lossColor(value,min,max){
  const t=clamp((Math.log1p(value)-Math.log1p(min))/(Math.log1p(max)-Math.log1p(min)||1),0,1);
  const r=Math.round(225-105*t),g=Math.round(243-83*t),b=Math.round(247-62*t);
  return `rgb(${r},${g},${b})`;
}

function lossSurfaceValues(cols,rows){
  const values=[];let min=Infinity,max=-Infinity;
  for(let row=0;row<rows;row++)for(let col=0;col<cols;col++){
    const w=-4+(col/(cols-1))*8,b=-5+(row/(rows-1))*10,value=mse(subset("train"),w,b);
    values.push({w,b,value,row,col});min=Math.min(min,value);max=Math.max(max,value);
  }
  return {values,min,max};
}
function normalizedLoss(value,min,max){return clamp((Math.log1p(value)-Math.log1p(min))/(Math.log1p(max)-Math.log1p(min)||1),0,1)}

function drawLossSurface2D(){
  const size=fitCanvas(lossCanvas,lossContext),area={x:40,y:16,w:size.width-54,h:size.height-48},context=lossContext;
  context.clearRect(0,0,size.width,size.height);
  const cols=48,rows=44,{values,min,max}=lossSurfaceValues(cols,rows);
  const cw=area.w/cols,ch=area.h/rows;
  values.forEach(point=>{const screenRow=rows-1-point.row;context.fillStyle=lossColor(point.value,min,max);context.fillRect(area.x+point.col*cw,area.y+screenRow*ch,cw+1,ch+1)});
  const lx=w=>area.x+(w+4)/8*area.w,ly=b=>area.y+(5-b)/10*area.h;
  context.strokeStyle="#ffffffbb";context.lineWidth=1;
  [-2,0,2].forEach(w=>{context.beginPath();context.moveTo(lx(w),area.y);context.lineTo(lx(w),area.y+area.h);context.stroke()});
  [-2,0,2].forEach(b=>{context.beginPath();context.moveTo(area.x,ly(b));context.lineTo(area.x+area.w,ly(b));context.stroke()});
  if(state.trajectory.length>1){
    context.beginPath();state.trajectory.forEach((point,index)=>{const x=lx(clamp(point.w,-4,4)),y=ly(clamp(point.b,-5,5));index?context.lineTo(x,y):context.moveTo(x,y)});context.strokeStyle="#ffcf70";context.lineWidth=3;context.stroke();
  }
  context.beginPath();context.arc(lx(clamp(state.w,-4,4)),ly(clamp(state.b,-5,5)),6,0,Math.PI*2);context.fillStyle="#c44b4b";context.fill();context.strokeStyle="#fff";context.lineWidth=2;context.stroke();
  context.strokeStyle="#91a4b4";context.lineWidth=1;context.strokeRect(area.x,area.y,area.w,area.h);
  context.fillStyle="#536477";context.font="600 12px system-ui";context.fillText("slope w",area.x+area.w-47,size.height-8);context.save();context.translate(12,area.y+54);context.rotate(-Math.PI/2);context.fillText("intercept b",0,0);context.restore();
  context.fillStyle="#69798a";context.font="11px system-ui";context.fillText("−4",area.x-4,size.height-15);context.fillText("4",area.x+area.w-3,size.height-15);context.fillText("5",area.x-18,area.y+4);context.fillText("−5",area.x-23,area.y+area.h+3);
}

function drawLossSurface3D(){
  const size=fitCanvas(lossCanvas,lossContext),context=lossContext,cols=27,rows=27,{values,min,max}=lossSurfaceValues(cols,rows);
  context.clearRect(0,0,size.width,size.height);context.fillStyle="#f8fbfd";context.fillRect(0,0,size.width,size.height);
  const scale=Math.min(size.width*.34,size.height*.34),heightScale=size.height*.43,centerX=size.width*.52,baseY=size.height*.68;
  const project=(w,b,z)=>{
    const x=w/4,y=b/5,cos=Math.cos(state.lossYaw),sin=Math.sin(state.lossYaw),screenGround=x*cos-y*sin,depth=x*sin+y*cos;
    return {x:centerX+screenGround*scale,y:baseY+depth*scale*state.lossTilt-z*heightScale,depth};
  };
  const pointAt=(row,col)=>values[row*cols+col];
  const cells=[];
  for(let row=0;row<rows-1;row++)for(let col=0;col<cols-1;col++){
    const corners=[pointAt(row,col),pointAt(row,col+1),pointAt(row+1,col+1),pointAt(row+1,col)];
    const projected=corners.map(point=>project(point.w,point.b,normalizedLoss(point.value,min,max)));
    cells.push({corners,projected,depth:projected.reduce((sum,point)=>sum+point.depth,0)/4,value:corners.reduce((sum,point)=>sum+point.value,0)/4});
  }
  cells.sort((a,b)=>a.depth-b.depth).forEach(cell=>{
    context.beginPath();cell.projected.forEach((point,index)=>index?context.lineTo(point.x,point.y):context.moveTo(point.x,point.y));context.closePath();
    context.fillStyle=lossColor(cell.value,min,max);context.fill();context.strokeStyle="#ffffff75";context.lineWidth=.55;context.stroke();
  });
  const drawAxis=(from,to,label,align="left")=>{const start=project(...from),end=project(...to);context.beginPath();context.moveTo(start.x,start.y);context.lineTo(end.x,end.y);context.strokeStyle="#42586d";context.lineWidth=1.5;context.stroke();context.fillStyle="#334b60";context.font="700 11px system-ui";context.textAlign=align;context.fillText(label,end.x+(align==="right"?-5:5),end.y-4)};
  drawAxis([-4,-5,0],[4,-5,0],"slope w");drawAxis([-4,-5,0],[-4,5,0],"intercept b","right");drawAxis([-4,-5,0],[-4,-5,1.15],"loss");context.textAlign="left";
  if(state.trajectory.length>1){
    context.beginPath();state.trajectory.forEach((point,index)=>{const value=mse(subset("train"),point.w,point.b),screen=project(point.w,point.b,normalizedLoss(value,min,max)+.025);index?context.lineTo(screen.x,screen.y):context.moveTo(screen.x,screen.y)});context.strokeStyle="#ffd06f";context.lineWidth=3.5;context.stroke();
  }
  const currentValue=mse(subset("train")),current=project(state.w,state.b,normalizedLoss(currentValue,min,max)+.03);
  context.beginPath();context.arc(current.x,current.y,6.5,0,Math.PI*2);context.fillStyle="#c44b4b";context.fill();context.strokeStyle="#fff";context.lineWidth=2;context.stroke();
  context.fillStyle="#657587";context.font="11px system-ui";context.fillText("log-scaled MSE height",12,18);
}

function drawLossSurface(){state.lossView==="3d"?drawLossSurface3D():drawLossSurface2D()}

function updateReadouts(){
  const grad=gradient();
  $("#slopeOut").textContent=fmt(state.w,2);$("#interceptOut").textContent=fmt(state.b,2);$("#learningRateOut").textContent=fmt(state.alpha,3);
  $("#trainMse").textContent=fmt(mse(subset("train")),3);$("#testMse").textContent=fmt(mse(subset("test")),3);
  $("#slopeGradient").textContent=fmt(grad.dw,3);$("#interceptGradient").textContent=fmt(grad.db,3);
  const point=state.points[state.selected];
  if(point){
    const yhat=prediction(point.x),residual=point.y-yhat;
    $("#pointCalculation").innerHTML=`For the selected point: ŷ = ${fmt(state.w,2)} × ${fmt(point.x,2)} + ${fmt(state.b,2)} = <b>${fmt(yhat,2)}</b>. Its residual is ${fmt(point.y,2)} − ${fmt(yhat,2)} = <b>${fmt(residual,2)}</b>.`;
  }
  $("#gradientCalculation").innerHTML=`Current gradient: ∂MSE/∂w = <b>${fmt(grad.dw,3)}</b>, ∂MSE/∂b = <b>${fmt(grad.db,3)}</b>. One step subtracts α times these values.`;
}

function render(){drawDataPlot();drawLossSurface();updateReadouts()}
function syncSliders(){$("#slope").value=String(state.w);$("#intercept").value=String(state.b);$("#learningRate").value=String(state.alpha)}
function setModel(w,b,record=true){state.w=clamp(w,-4,4);state.b=clamp(b,-5,5);if(record)state.trajectory.push({w:state.w,b:state.b});syncSliders();render()}
function gradientStep(){
  const {dw,db}=gradient();setModel(state.w-state.alpha*dw,state.b-state.alpha*db,true);state.autoSteps++;
  if(state.autoSteps>=60||Math.hypot(dw,db)<.0005)stopAutoStep();
}
function stopAutoStep(){if(state.autoTimer)clearInterval(state.autoTimer);state.autoTimer=null;$("#playButton").textContent="Auto-step"}

$("#stepButton").addEventListener("click",()=>{stopAutoStep();gradientStep()});
$("#playButton").addEventListener("click",()=>{
  if(state.autoTimer){stopAutoStep();return}
  state.autoSteps=0;$("#playButton").textContent="Pause";state.autoTimer=setInterval(gradientStep,window.labDelay?window.labDelay(180):180);
});
$("#bestFitButton").addEventListener("click",()=>{
  stopAutoStep();const points=subset("train"),meanX=points.reduce((s,p)=>s+p.x,0)/points.length,meanY=points.reduce((s,p)=>s+p.y,0)/points.length;
  const numerator=points.reduce((s,p)=>s+(p.x-meanX)*(p.y-meanY),0),denominator=points.reduce((s,p)=>s+(p.x-meanX)**2,0);
  const bestW=denominator<1e-9?0:numerator/denominator;
  setModel(bestW,meanY-bestW*meanX,true);
});
$("#resetModelButton").addEventListener("click",()=>{stopAutoStep();state.trajectory=[{w:0,b:0}];setModel(0,0,false)});
$("#datasetPreset").addEventListener("change",event=>{stopAutoStep();state.points=buildDataset(event.target.value);state.selected=2;state.trajectory=[{w:0,b:0}];setModel(0,0,false)});
$("#slope").addEventListener("input",event=>{stopAutoStep();state.w=Number(event.target.value);state.trajectory=[{w:state.w,b:state.b}];render()});
$("#intercept").addEventListener("input",event=>{stopAutoStep();state.b=Number(event.target.value);state.trajectory=[{w:state.w,b:state.b}];render()});
$("#learningRate").addEventListener("input",event=>{state.alpha=Number(event.target.value);updateReadouts()});
$("#showResiduals").addEventListener("change",render);$("#showTest").addEventListener("change",render);

function setLossView(view){
  state.lossView=view;const is3d=view==="3d";
  $("#loss3dButton").classList.toggle("active",is3d);$("#loss3dButton").setAttribute("aria-pressed",String(is3d));
  $("#loss2dButton").classList.toggle("active",!is3d);$("#loss2dButton").setAttribute("aria-pressed",String(!is3d));
  lossCanvas.style.cursor=is3d?"grab":"default";
  lossCanvas.setAttribute("aria-label",is3d?"Interactive three-dimensional loss surface showing the current model and gradient descent path":"Two-dimensional loss contours showing the current model and gradient descent path");
  $("#lossHelp").textContent=is3d?"Drag or use the arrow keys to rotate. Height represents training loss; the red marker is the current model and the yellow line records gradient-descent steps.":"Each location represents one possible line. Darker regions have higher error; the yellow path records gradient-descent steps.";
  drawLossSurface();
}
$("#loss3dButton").addEventListener("click",()=>setLossView("3d"));
$("#loss2dButton").addEventListener("click",()=>setLossView("2d"));
lossCanvas.addEventListener("pointerdown",event=>{if(state.lossView!=="3d")return;state.lossDragging=true;state.lossPointer={x:event.clientX,y:event.clientY};lossCanvas.setPointerCapture(event.pointerId)});
lossCanvas.addEventListener("pointermove",event=>{if(!state.lossDragging||state.lossView!=="3d")return;const dx=event.clientX-state.lossPointer.x,dy=event.clientY-state.lossPointer.y;state.lossYaw+=dx*.012;state.lossTilt=clamp(state.lossTilt+dy*.004,.16,.72);state.lossPointer={x:event.clientX,y:event.clientY};drawLossSurface()});
function endLossDrag(){state.lossDragging=false}lossCanvas.addEventListener("pointerup",endLossDrag);lossCanvas.addEventListener("pointercancel",endLossDrag);
lossCanvas.addEventListener("dblclick",()=>{state.lossYaw=.72;state.lossTilt=.43;drawLossSurface()});
lossCanvas.addEventListener("keydown",event=>{if(state.lossView!=="3d"||!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(event.key))return;event.preventDefault();if(event.key==="ArrowLeft")state.lossYaw-=.12;if(event.key==="ArrowRight")state.lossYaw+=.12;if(event.key==="ArrowUp")state.lossTilt=clamp(state.lossTilt-.05,.16,.72);if(event.key==="ArrowDown")state.lossTilt=clamp(state.lossTilt+.05,.16,.72);drawLossSurface()});

function pointerPosition(event){const rect=dataCanvas.getBoundingClientRect();return{x:event.clientX-rect.left,y:event.clientY-rect.top}}
function nearestPoint(position){
  const size={width:dataCanvas.getBoundingClientRect().width,height:dataCanvas.getBoundingClientRect().height},area=plotArea(size);let best=-1,distance=Infinity;
  state.points.forEach((point,index)=>{const d=Math.hypot(position.x-xPixel(point.x,area),position.y-yPixel(point.y,area));if(d<distance){distance=d;best=index}});
  return distance<14?best:-1;
}
dataCanvas.addEventListener("pointerdown",event=>{
  stopAutoStep();const position=pointerPosition(event),size={width:dataCanvas.getBoundingClientRect().width,height:dataCanvas.getBoundingClientRect().height},area=plotArea(size),nearest=nearestPoint(position);
  if(event.shiftKey&&nearest>=0){
    const target=state.points[nearest],canRemove=target.split==="test"||subset("train").length>2;
    if(canRemove){state.points.splice(nearest,1);state.selected=clamp(state.selected,0,state.points.length-1)}
    render();return
  }
  if(nearest>=0){state.dragging=nearest;state.selected=nearest}else if(position.x>=area.x&&position.x<=area.x+area.w&&position.y>=area.y&&position.y<=area.y+area.h){
    state.points.push({x:clamp(xValue(position.x,area),-4.8,4.8),y:clamp(yValue(position.y,area),-4.8,4.8),split:"train"});state.selected=state.points.length-1;state.dragging=state.selected;
  }
  dataCanvas.setPointerCapture(event.pointerId);render();
});
dataCanvas.addEventListener("pointermove",event=>{
  if(state.dragging<0)return;const position=pointerPosition(event),size={width:dataCanvas.getBoundingClientRect().width,height:dataCanvas.getBoundingClientRect().height},area=plotArea(size),point=state.points[state.dragging];
  point.x=clamp(xValue(position.x,area),-4.8,4.8);point.y=clamp(yValue(position.y,area),-4.8,4.8);state.trajectory=[{w:state.w,b:state.b}];render();
});
function endDrag(){state.dragging=-1}dataCanvas.addEventListener("pointerup",endDrag);dataCanvas.addEventListener("pointercancel",endDrag);

document.querySelectorAll(".tab").forEach(tab=>tab.addEventListener("click",()=>{
  document.querySelectorAll(".tab").forEach(item=>{const active=item===tab;item.classList.toggle("active",active);item.setAttribute("aria-selected",String(active))});
  document.querySelectorAll(".tab-panel").forEach(panel=>{const active=panel.id===`tab-${tab.dataset.tab}`;panel.classList.toggle("active",active);panel.hidden=!active});
}));
document.querySelectorAll("[data-emphasis]").forEach(term=>{
  term.addEventListener("mouseenter",()=>{state.emphasis=term.dataset.emphasis;term.classList.add("active");drawDataPlot()});
  term.addEventListener("mouseleave",()=>{state.emphasis="";term.classList.remove("active");drawDataPlot()});
});

function buildLogDataset(type){
  const random=seededRandom({separable:118,overlap:227,imbalanced:349,nonlinear:463}[type]||118);
  const points=[];
  const jitter=amount=>(random()-.5)*2*amount;
  if(type==="nonlinear"){
    for(let i=0;i<12;i++){
      const angle=i/12*Math.PI*2+jitter(.13),radius=.65+random()*.75;
      points.push({x:Math.cos(angle)*radius,y:Math.sin(angle)*radius,label:1});
    }
    for(let i=0;i<20;i++){
      const angle=i/20*Math.PI*2+jitter(.11),radius=2.65+random()*1.1;
      points.push({x:Math.cos(angle)*radius,y:Math.sin(angle)*radius,label:0});
    }
    return points;
  }
  const countZero=type==="imbalanced"?23:15,countOne=type==="imbalanced"?7:15;
  const spread=type==="overlap"?1.5:.82;
  const zeroCenter=type==="overlap"?[-.8,-.45]:[-1.75,-.9];
  const oneCenter=type==="overlap"?[.8,.45]:[1.75,.9];
  for(let i=0;i<countZero;i++)points.push({x:clamp(zeroCenter[0]+jitter(spread),-4.6,4.6),y:clamp(zeroCenter[1]+jitter(spread),-4.6,4.6),label:0});
  for(let i=0;i<countOne;i++)points.push({x:clamp(oneCenter[0]+jitter(spread),-4.6,4.6),y:clamp(oneCenter[1]+jitter(spread),-4.6,4.6),label:1});
  if(type==="separable"){
    points[3]={x:.45,y:.25,label:0};
    points[countZero+4]={x:-.35,y:.65,label:1};
  }
  return points;
}

const logState={
  points:buildLogDataset("separable"),w1:0,w2:0,b:0,threshold:.5,alpha:.08,
  selected:2,dragging:-1,autoTimer:null,autoSteps:0
};
const logisticCanvas=$("#logisticCanvas"),sigmoidCanvas=$("#sigmoidCanvas");
const logisticContext=logisticCanvas.getContext("2d"),sigmoidContext=sigmoidCanvas.getContext("2d");

function sigmoid(value){return 1/(1+Math.exp(-clamp(value,-30,30)))}
function logScore(point){return logState.w1*point.x+logState.w2*point.y+logState.b}
function logProbability(point){return sigmoid(logScore(point))}
function logit(probability){return Math.log(probability/(1-probability))}
function logLoss(){
  if(!logState.points.length)return NaN;
  return logState.points.reduce((sum,point)=>{
    const probability=clamp(logProbability(point),1e-9,1-1e-9);
    return sum-(point.label*Math.log(probability)+(1-point.label)*Math.log(1-probability));
  },0)/logState.points.length;
}
function logGradient(){
  if(!logState.points.length)return {dw1:0,dw2:0,db:0};
  let dw1=0,dw2=0,db=0;
  logState.points.forEach(point=>{const error=logProbability(point)-point.label;dw1+=error*point.x;dw2+=error*point.y;db+=error});
  const count=logState.points.length;
  return {dw1:dw1/count,dw2:dw2/count,db:db/count};
}
function logMetrics(){
  const counts={tn:0,fp:0,fn:0,tp:0};
  logState.points.forEach(point=>{
    const predicted=logProbability(point)>=logState.threshold?1:0;
    if(point.label===0&&predicted===0)counts.tn++;
    if(point.label===0&&predicted===1)counts.fp++;
    if(point.label===1&&predicted===0)counts.fn++;
    if(point.label===1&&predicted===1)counts.tp++;
  });
  const total=logState.points.length;
  return {...counts,accuracy:total?(counts.tp+counts.tn)/total:0,precision:counts.tp+counts.fp?counts.tp/(counts.tp+counts.fp):0,recall:counts.tp+counts.fn?counts.tp/(counts.tp+counts.fn):0};
}

function logDrawGrid(context,area){
  context.save();context.strokeStyle="#dfe7ed";context.lineWidth=1;context.fillStyle="#657587";context.font="12px system-ui";
  for(let value=-4;value<=4;value+=2){
    const xp=xPixel(value,area),yp=yPixel(value,area);
    context.beginPath();context.moveTo(xp,area.y);context.lineTo(xp,area.y+area.h);context.stroke();
    context.beginPath();context.moveTo(area.x,yp);context.lineTo(area.x+area.w,yp);context.stroke();
    context.fillText(String(value),xp-5,area.y+area.h+19);context.fillText(String(value),area.x-26,yp+4);
  }
  context.strokeStyle="#92a3b0";context.lineWidth=1.3;
  context.beginPath();context.moveTo(xPixel(0,area),area.y);context.lineTo(xPixel(0,area),area.y+area.h);context.stroke();
  context.beginPath();context.moveTo(area.x,yPixel(0,area));context.lineTo(area.x+area.w,yPixel(0,area));context.stroke();
  context.fillStyle="#536477";context.font="600 12px system-ui";context.fillText("x₁",area.x+area.w-7,area.y+area.h+30);context.fillText("x₂",area.x-37,area.y+9);context.restore();
}
function probabilityColor(probability){
  const low=[222,237,248],high=[251,229,213];
  const values=low.map((value,index)=>Math.round(value+(high[index]-value)*probability));
  return `rgb(${values[0]},${values[1]},${values[2]})`;
}
function boundaryIntersections(){
  const target=logit(logState.threshold)-logState.b,points=[],epsilon=1e-8;
  if(Math.abs(logState.w2)>epsilon){
    [RANGE.min,RANGE.max].forEach(x=>{const y=(target-logState.w1*x)/logState.w2;if(y>=RANGE.min&&y<=RANGE.max)points.push({x,y})});
  }
  if(Math.abs(logState.w1)>epsilon){
    [RANGE.min,RANGE.max].forEach(y=>{const x=(target-logState.w2*y)/logState.w1;if(x>=RANGE.min&&x<=RANGE.max&&!points.some(point=>Math.hypot(point.x-x,point.y-y)<1e-5))points.push({x,y})});
  }
  return points.slice(0,2);
}
function drawLogisticPlot(){
  const size=fitCanvas(logisticCanvas,logisticContext),area=plotArea(size),context=logisticContext;
  context.clearRect(0,0,size.width,size.height);context.save();context.beginPath();context.rect(area.x,area.y,area.w,area.h);context.clip();
  if($("#logShowProbability").checked){
    const cells=34,cellWidth=area.w/cells,cellHeight=area.h/cells;
    for(let row=0;row<cells;row++)for(let column=0;column<cells;column++){
      const point={x:RANGE.min+(column+.5)/cells*10,y:RANGE.max-(row+.5)/cells*10};
      context.fillStyle=probabilityColor(logProbability(point));context.fillRect(area.x+column*cellWidth,area.y+row*cellHeight,cellWidth+1,cellHeight+1);
    }
  }
  context.restore();logDrawGrid(context,area);
  context.save();context.beginPath();context.rect(area.x,area.y,area.w,area.h);context.clip();
  const intersections=boundaryIntersections();
  if(intersections.length===2){
    context.beginPath();context.moveTo(xPixel(intersections[0].x,area),yPixel(intersections[0].y,area));context.lineTo(xPixel(intersections[1].x,area),yPixel(intersections[1].y,area));context.strokeStyle="#17233a";context.lineWidth=3;context.stroke();
  }
  logState.points.forEach((point,index)=>{
    const x=xPixel(point.x,area),y=yPixel(point.y,area),selected=index===logState.selected;
    context.beginPath();context.arc(x,y,selected?7:5.6,0,Math.PI*2);context.fillStyle=point.label===1?"#e27835":"#246eaa";context.fill();context.strokeStyle="#fff";context.lineWidth=2;context.stroke();
    if(selected){context.beginPath();context.arc(x,y,10,0,Math.PI*2);context.strokeStyle="#17233a";context.lineWidth=2;context.stroke()}
    const predicted=logProbability(point)>=logState.threshold?1:0;
    if(predicted!==point.label){context.beginPath();context.moveTo(x-4,y-4);context.lineTo(x+4,y+4);context.moveTo(x+4,y-4);context.lineTo(x-4,y+4);context.strokeStyle="#5b2630";context.lineWidth=1.7;context.stroke()}
  });
  context.restore();
}

function drawSigmoid(){
  const size=fitCanvas(sigmoidCanvas,sigmoidContext),context=sigmoidContext,area={x:38,y:12,w:size.width-52,h:size.height-42};
  context.clearRect(0,0,size.width,size.height);context.fillStyle="#fbfdfe";context.fillRect(area.x,area.y,area.w,area.h);
  const sx=value=>area.x+(value+6)/12*area.w,sy=value=>area.y+(1-value)*area.h;
  context.strokeStyle="#dce5ec";context.lineWidth=1;[0,.5,1].forEach(value=>{context.beginPath();context.moveTo(area.x,sy(value));context.lineTo(area.x+area.w,sy(value));context.stroke()});
  context.strokeStyle="#17233a";context.lineWidth=1.2;context.strokeRect(area.x,area.y,area.w,area.h);
  context.setLineDash([5,4]);context.beginPath();context.moveTo(area.x,sy(logState.threshold));context.lineTo(area.x+area.w,sy(logState.threshold));context.strokeStyle="#9a6538";context.stroke();context.setLineDash([]);
  context.beginPath();for(let i=0;i<=120;i++){const z=-6+i/10,x=sx(z),y=sy(sigmoid(z));i?context.lineTo(x,y):context.moveTo(x,y)}context.strokeStyle="#176ca4";context.lineWidth=3;context.stroke();
  const point=logState.points[logState.selected];
  if(point){const z=clamp(logScore(point),-6,6),probability=logProbability(point),x=sx(z),y=sy(probability);context.beginPath();context.moveTo(x,sy(0));context.lineTo(x,y);context.strokeStyle="#e27835";context.lineWidth=1.5;context.stroke();context.beginPath();context.arc(x,y,5,0,Math.PI*2);context.fillStyle="#e27835";context.fill()}
  context.fillStyle="#607184";context.font="11px system-ui";context.fillText("0",area.x-18,sy(0)+4);context.fillText("0.5",area.x-29,sy(.5)+4);context.fillText("1",area.x-18,sy(1)+4);context.fillText("score z",area.x+area.w-38,size.height-7);context.fillText("−6",area.x-5,size.height-8);context.fillText("6",area.x+area.w-4,size.height-8);
}

function updateLogReadouts(){
  const metrics=logMetrics(),point=logState.points[logState.selected];
  $("#logW1Out").textContent=fmt(logState.w1,2);$("#logW2Out").textContent=fmt(logState.w2,2);$("#logBiasOut").textContent=fmt(logState.b,2);$("#logThresholdOut").textContent=fmt(logState.threshold,2);
  $("#logLoss").textContent=fmt(logLoss(),3);$("#logAccuracy").textContent=`${Math.round(metrics.accuracy*100)}%`;$("#logPrecision").textContent=`${Math.round(metrics.precision*100)}%`;$("#logRecall").textContent=`${Math.round(metrics.recall*100)}%`;
  $("#logTN").textContent=metrics.tn;$("#logFP").textContent=metrics.fp;$("#logFN").textContent=metrics.fn;$("#logTP").textContent=metrics.tp;$("#logThresholdFormula").textContent=fmt(logState.threshold,2);
  const target=logit(logState.threshold);
  $("#logBoundaryCalculation").innerHTML=`The boundary contains points where <b>${fmt(logState.w1,2)}x₁ + ${fmt(logState.w2,2)}x₂ + ${fmt(logState.b,2)} = ${fmt(target,2)}</b>, the score corresponding to the current threshold.`;
  if(point){
    const score=logScore(point),probability=logProbability(point),prediction=probability>=logState.threshold?1:0;
    $("#selectedProbability").innerHTML=`Selected point: score <b>${fmt(score,2)}</b> → Class 1 probability <b>${fmt(probability*100,1)}%</b> → predicted Class <b>${prediction}</b>.`;
    $("#logPointCalculation").innerHTML=`For the selected point: z = ${fmt(logState.w1,2)}(${fmt(point.x,2)}) + ${fmt(logState.w2,2)}(${fmt(point.y,2)}) + ${fmt(logState.b,2)} = <b>${fmt(score,2)}</b>. The sigmoid gives p = <b>${fmt(probability,3)}</b>.`;
  }
}
function logRender(){drawLogisticPlot();drawSigmoid();updateLogReadouts()}
function logSyncSliders(){$("#logW1").value=String(logState.w1);$("#logW2").value=String(logState.w2);$("#logBias").value=String(logState.b);$("#logThreshold").value=String(logState.threshold)}
function logResetModel(){logState.w1=0;logState.w2=0;logState.b=0;logState.autoSteps=0;logSyncSliders();logRender()}
function logGradientStep(){
  const {dw1,dw2,db}=logGradient();logState.w1=clamp(logState.w1-logState.alpha*dw1,-4,4);logState.w2=clamp(logState.w2-logState.alpha*dw2,-4,4);logState.b=clamp(logState.b-logState.alpha*db,-5,5);logState.autoSteps++;logSyncSliders();logRender();
  if(logState.autoSteps>=80||Math.hypot(dw1,dw2,db)<.0008)logStopAutoStep();
}
function logStopAutoStep(){if(logState.autoTimer)clearInterval(logState.autoTimer);logState.autoTimer=null;const button=$("#logPlayButton");if(button)button.textContent="Auto-step"}

$("#logStepButton").addEventListener("click",()=>{logStopAutoStep();logGradientStep()});
$("#logPlayButton").addEventListener("click",()=>{if(logState.autoTimer){logStopAutoStep();return}logState.autoSteps=0;$("#logPlayButton").textContent="Pause";logState.autoTimer=setInterval(logGradientStep,window.labDelay?window.labDelay(150):150)});
$("#logResetButton").addEventListener("click",()=>{logStopAutoStep();logResetModel()});
$("#logDatasetPreset").addEventListener("change",event=>{logStopAutoStep();logState.points=buildLogDataset(event.target.value);logState.selected=2;logResetModel()});
[["#logW1","w1"],["#logW2","w2"],["#logBias","b"]].forEach(([selector,key])=>$(selector).addEventListener("input",event=>{logStopAutoStep();logState[key]=Number(event.target.value);logRender()}));
$("#logThreshold").addEventListener("input",event=>{logState.threshold=Number(event.target.value);logRender()});
$("#logShowProbability").addEventListener("change",logRender);

function logPointerPosition(event){const rect=logisticCanvas.getBoundingClientRect();return {x:event.clientX-rect.left,y:event.clientY-rect.top}}
function logNearestPoint(position){
  const size={width:logisticCanvas.getBoundingClientRect().width,height:logisticCanvas.getBoundingClientRect().height},area=plotArea(size);let best=-1,distance=Infinity;
  logState.points.forEach((point,index)=>{const candidate=Math.hypot(position.x-xPixel(point.x,area),position.y-yPixel(point.y,area));if(candidate<distance){distance=candidate;best=index}});return distance<14?best:-1;
}
logisticCanvas.addEventListener("pointerdown",event=>{
  logStopAutoStep();const position=logPointerPosition(event),size={width:logisticCanvas.getBoundingClientRect().width,height:logisticCanvas.getBoundingClientRect().height},area=plotArea(size),nearest=logNearestPoint(position);
  if(event.shiftKey&&nearest>=0){if(logState.points.length>2){logState.points.splice(nearest,1);logState.selected=clamp(logState.selected,0,logState.points.length-1)}logRender();return}
  if(nearest>=0){logState.dragging=nearest;logState.selected=nearest}else if(position.x>=area.x&&position.x<=area.x+area.w&&position.y>=area.y&&position.y<=area.y+area.h){
    const label=Number(document.querySelector('input[name="logAddClass"]:checked').value);logState.points.push({x:clamp(xValue(position.x,area),-4.8,4.8),y:clamp(yValue(position.y,area),-4.8,4.8),label});logState.selected=logState.points.length-1;logState.dragging=logState.selected;
  }
  logisticCanvas.setPointerCapture(event.pointerId);logRender();
});
logisticCanvas.addEventListener("pointermove",event=>{
  if(logState.dragging<0)return;const position=logPointerPosition(event),size={width:logisticCanvas.getBoundingClientRect().width,height:logisticCanvas.getBoundingClientRect().height},area=plotArea(size),point=logState.points[logState.dragging];point.x=clamp(xValue(position.x,area),-4.8,4.8);point.y=clamp(yValue(position.y,area),-4.8,4.8);logRender();
});
function logEndDrag(){logState.dragging=-1}logisticCanvas.addEventListener("pointerup",logEndDrag);logisticCanvas.addEventListener("pointercancel",logEndDrag);

document.querySelectorAll(".log-tab").forEach(tab=>tab.addEventListener("click",()=>{
  document.querySelectorAll(".log-tab").forEach(item=>{const active=item===tab;item.classList.toggle("active",active);item.setAttribute("aria-selected",String(active))});
  document.querySelectorAll(".log-tab-panel").forEach(panel=>{const active=panel.id===`log-tab-${tab.dataset.logTab}`;panel.classList.toggle("active",active);panel.hidden=!active});
}));

window.addEventListener("resize",()=>requestAnimationFrame(()=>{$("#linearModule").hidden||render();$("#logisticModule").hidden||logRender()}));
syncSliders();logSyncSliders();
if(new URLSearchParams(window.location.search).get("loss")==="2d")setLossView("2d");
render();
