"use strict";

CONCEPT_META.training={
  category:"Foundations",
  title:"Neural Network Playground",
  summary:"Build and train a small neural classifier while watching features, neurons, connections and the decision boundary change.",
  question:"How do simple neurons combine features into a nonlinear decision?"
};

const TRAINING_FEATURES={
  x:{label:"x₁",long:"horizontal coordinate",value:point=>point.x,map:point=>point.x/1.2},
  y:{label:"x₂",long:"vertical coordinate",value:point=>point.y,map:point=>point.y/1.2},
  xy:{label:"x₁x₂",long:"feature interaction",value:point=>point.x*point.y,map:point=>point.x*point.y/1.44},
  x2:{label:"x₁²",long:"squared horizontal coordinate",value:point=>point.x*point.x,map:point=>(point.x*point.x-.72)/.72},
  y2:{label:"x₂²",long:"squared vertical coordinate",value:point=>point.y*point.y,map:point=>(point.y*point.y-.72)/.72}
};

function trainingFeatureVector(point,featureIds){return featureIds.map(id=>TRAINING_FEATURES[id].value(point))}
function trainingActivation(name,value){
  if(name==="relu")return Math.max(0,value);
  if(name==="sigmoid")return 1/(1+Math.exp(-clamp(value,-18,18)));
  return Math.tanh(value);
}
function trainingActivationDerivative(name,z,activation){
  if(name==="relu")return z>0?1:0;
  if(name==="sigmoid")return activation*(1-activation);
  return 1-activation*activation;
}

function trainingDataset(type,total,noise,trainRatio,seed){
  const random=seededRandom(seed),points=[],jitter=()=>((random()+random()+random())-1.5)*noise*.38;
  for(let index=0;index<total;index++){
    let x=0,y=0,label=index%2;const classIndex=Math.floor(index/2),perClass=Math.ceil(total/2);
    if(type==="circles"){
      const outer=label===1,angle=(classIndex+random()*.35)/perClass*Math.PI*2,radius=(outer?.78:.24)+(random()-.5)*(outer?.22:.2);
      x=radius*Math.cos(angle);y=radius*Math.sin(angle);
    }else if(type==="moons"){
      const angle=(classIndex+random()*.35)/Math.max(1,perClass-1)*Math.PI;
      if(label===0){x=-.33+.72*Math.cos(angle);y=.06+.68*Math.sin(angle)}
      else{x=.33-.72*Math.cos(angle);y=-.06-.68*Math.sin(angle)}
    }else if(type==="gaussian"){
      const mean=label===1?.48:-.48;x=mean+jitter()*2.2;y=mean+jitter()*2.2;
    }else{
      x=random()*2-1;y=random()*2-1;label=x*y>0?1:0;
    }
    x+=jitter();y+=jitter();if(random()<noise*.1)label=1-label;
    points.push({x:clamp(x,-1.2,1.2),y:clamp(y,-1.2,1.2),label,split:"train"});
  }
  for(let index=points.length-1;index>0;index--){const target=Math.floor(random()*(index+1));[points[index],points[target]]=[points[target],points[index]]}
  const trainCount=clamp(Math.round(points.length*trainRatio),2,points.length-2);
  points.forEach((point,index)=>point.split=index<trainCount?"train":"validation");
  return points;
}

function trainingCreateModel(inputSize,widths,seed){
  const random=seededRandom(seed),sizes=[inputSize,...widths,1],weights=[],biases=[];
  for(let layer=0;layer<sizes.length-1;layer++){
    const scale=Math.sqrt(6/(sizes[layer]+sizes[layer+1]));
    weights.push(Array.from({length:sizes[layer+1]},()=>Array.from({length:sizes[layer]},()=>((random()*2)-1)*scale)));
    biases.push(Array(sizes[layer+1]).fill(0));
  }
  return {weights,biases};
}
function trainingClone(model){return {weights:model.weights.map(layer=>layer.map(row=>[...row])),biases:model.biases.map(layer=>[...layer])}}
function trainingForward(model,input,activationName="tanh",dropout=0,random=null,isTraining=false){
  let activation=[...input];const caches=[];
  model.weights.forEach((weights,layer)=>{
    const z=weights.map((row,index)=>row.reduce((sum,weight,inputIndex)=>sum+weight*activation[inputIndex],model.biases[layer][index]));
    const last=layer===model.weights.length-1;
    const raw=last?z.map(value=>1/(1+Math.exp(-clamp(value,-18,18)))):z.map(value=>trainingActivation(activationName,value));
    const mask=raw.map(()=>!last&&isTraining&&dropout>0?(random()>=dropout?1/(1-dropout):0):1);
    const output=raw.map((value,index)=>value*mask[index]);
    caches.push({input:[...activation],z,raw,mask,output});activation=output;
  });
  return {probability:activation[0],caches};
}
function trainingPenalty(model,type,strength){
  if(type==="none"||strength===0)return 0;let total=0;
  model.weights.forEach(layer=>layer.forEach(row=>row.forEach(weight=>total+=type==="l1"?Math.abs(weight):weight*weight)));
  return strength*total;
}
function trainingEvaluate(model,points,settings){
  let loss=0,correct=0;
  points.forEach(point=>{
    const input=trainingFeatureVector(point,settings.features);
    const probability=clamp(trainingForward(model,input,settings.activation).probability,1e-7,1-1e-7);
    loss-=point.label*Math.log(probability)+(1-point.label)*Math.log(1-probability);
    if((probability>=.5?1:0)===point.label)correct++;
  });
  const mean=loss/Math.max(1,points.length);
  return {loss:mean,objective:mean+trainingPenalty(model,settings.regularizer,settings.strength),accuracy:correct/Math.max(1,points.length)};
}
function trainingGradientStep(model,points,settings,seed){
  const weightGradients=model.weights.map(layer=>layer.map(row=>row.map(()=>0)));
  const biasGradients=model.biases.map(layer=>layer.map(()=>0)),random=seededRandom(seed);
  points.forEach(point=>{
    const input=trainingFeatureVector(point,settings.features);
    const forward=trainingForward(model,input,settings.activation,settings.dropout,random,true),caches=forward.caches;
    let delta=[forward.probability-point.label];
    for(let layer=model.weights.length-1;layer>=0;layer--){
      const cache=caches[layer];
      for(let output=0;output<delta.length;output++){
        biasGradients[layer][output]+=delta[output];
        for(let inputIndex=0;inputIndex<cache.input.length;inputIndex++)weightGradients[layer][output][inputIndex]+=delta[output]*cache.input[inputIndex];
      }
      if(layer>0){
        const previous=caches[layer-1],nextDelta=Array(previous.raw.length).fill(0);
        for(let inputIndex=0;inputIndex<nextDelta.length;inputIndex++){
          let upstream=0;
          for(let output=0;output<delta.length;output++)upstream+=model.weights[layer][output][inputIndex]*delta[output];
          nextDelta[inputIndex]=upstream*trainingActivationDerivative(settings.activation,previous.z[inputIndex],previous.raw[inputIndex])*previous.mask[inputIndex];
        }
        delta=nextDelta;
      }
    }
  });
  const count=Math.max(1,points.length);
  model.weights.forEach((layer,layerIndex)=>layer.forEach((row,output)=>row.forEach((weight,inputIndex)=>{
    let gradient=weightGradients[layerIndex][output][inputIndex]/count;
    if(settings.regularizer==="l2")gradient+=settings.strength*2*weight;
    if(settings.regularizer==="l1")gradient+=settings.strength*Math.sign(weight);
    model.weights[layerIndex][output][inputIndex]-=settings.rate*clamp(gradient,-5,5);
  })));
  model.biases.forEach((layer,layerIndex)=>layer.forEach((bias,index)=>model.biases[layerIndex][index]-=settings.rate*clamp(biasGradients[layerIndex][index]/count,-5,5)));
}

function trainingDatasetGlyph(type){
  if(type==="circles")return '<svg viewBox="0 0 54 38" aria-hidden="true"><circle cx="27" cy="19" r="7" fill="#e27835"/><g fill="#246eaa"><circle cx="27" cy="4" r="3"/><circle cx="40" cy="10" r="3"/><circle cx="43" cy="25" r="3"/><circle cx="27" cy="34" r="3"/><circle cx="12" cy="27" r="3"/><circle cx="12" cy="11" r="3"/></g></svg>';
  if(type==="xor")return '<svg viewBox="0 0 54 38" aria-hidden="true"><g fill="#246eaa"><circle cx="13" cy="10" r="4"/><circle cx="40" cy="28" r="4"/></g><g fill="#e27835"><circle cx="40" cy="10" r="4"/><circle cx="13" cy="28" r="4"/></g></svg>';
  if(type==="moons")return '<svg viewBox="0 0 54 38" aria-hidden="true"><path d="M8 23 Q20 2 34 18" fill="none" stroke="#e27835" stroke-width="5" stroke-linecap="round"/><path d="M20 21 Q34 38 47 16" fill="none" stroke="#246eaa" stroke-width="5" stroke-linecap="round"/></svg>';
  return '<svg viewBox="0 0 54 38" aria-hidden="true"><g fill="#e27835"><circle cx="14" cy="12" r="4"/><circle cx="20" cy="18" r="3"/><circle cx="10" cy="22" r="3"/></g><g fill="#246eaa"><circle cx="40" cy="25" r="4"/><circle cx="34" cy="20" r="3"/><circle cx="44" cy="16" r="3"/></g></svg>';
}
function trainingDatasetOption(type,label,checked=false){return `<label class="train-dataset-option"><input type="radio" name="trainDataset" id="trainDataset${type[0].toUpperCase()+type.slice(1)}" value="${type}"${checked?" checked":""}><span class="train-dataset-card">${trainingDatasetGlyph(type)}<b>${label}</b></span></label>`}

function setupTraining(){
  setConcept(
    CONCEPT_META.training,
    "Interactive network",
    "Features → hidden patterns → prediction",
    "Press Train or Step. Click any feature, neuron or output map to inspect what it represents. Line colour shows sign; thickness shows weight magnitude.",
    "<strong>Start here</strong><p>Press <b>Train</b> and watch the output boundary form. Then click hidden neurons: each square shows the pattern that neuron responds to across the whole input space.</p>"
  );
  conceptElements.controls.innerHTML=
    '<span class="field-label">Visualization</span><div class="train-view-switch" role="radiogroup" aria-label="Training visualization"><label><input id="trainViewNetwork" type="radio" name="trainView" value="network" checked><span>Network</span></label><label><input id="trainViewCurves" type="radio" name="trainView" value="curves"><span>Curves &amp; A/B</span></label></div>'+
    '<div class="train-control-section"><span class="train-control-heading">1 · Data</span><span class="field-label">Dataset</span><div class="train-dataset-picker">'+
      trainingDatasetOption("circles","Circle",true)+trainingDatasetOption("xor","XOR")+trainingDatasetOption("moons","Moons")+trainingDatasetOption("gaussian","Groups")+
    '</div>'+rangeControl("trainSamples","Total examples","n",40,160,20,100)+rangeControl("trainRatio","Training share","%",20,80,5,50)+rangeControl("trainNoise","Noise","σ",0,.5,.05,0)+'</div>'+
    '<div class="train-control-section"><span class="train-control-heading">2 · Features</span><div class="train-feature-grid">'+
      '<label><input id="trainFeatureX" type="checkbox" value="x" checked><span>x₁</span></label><label><input id="trainFeatureY" type="checkbox" value="y" checked><span>x₂</span></label><label><input id="trainFeatureXY" type="checkbox" value="xy"><span>x₁x₂</span></label><label><input id="trainFeatureX2" type="checkbox" value="x2"><span>x₁²</span></label><label><input id="trainFeatureY2" type="checkbox" value="y2"><span>x₂²</span></label></div></div>'+
    '<div class="train-control-section"><span class="train-control-heading">3 · Network</span>'+selectControl("trainActivation","Activation",[["tanh","Tanh"],["relu","ReLU"],["sigmoid","Sigmoid"]])+selectControl("trainDepth","Hidden layers",[["1","One hidden layer"],["2","Two hidden layers"]])+rangeControl("trainWidth1","First-layer neurons","H₁",1,6,1,4)+'<div id="trainSecondLayer">'+rangeControl("trainWidth2","Second-layer neurons","H₂",1,6,1,2)+'</div></div>'+
    '<div class="train-control-section"><span class="train-control-heading">4 · Learning</span>'+selectControl("trainRate","Learning rate",[["0.001","0.001"],["0.003","0.003"],["0.01","0.01"],["0.03","0.03"],["0.1","0.1"],["0.3","0.3"]])+selectControl("trainBatch","Batch size",[["1","1"],["5","5"],["10","10"],["20","20"],["all","All training examples"]])+selectControl("trainRegularizer","Weight penalty",[["none","None"],["l1","L1"],["l2","L2"]])+rangeControl("trainStrength","Regularization rate","λ",0,.05,.005,0)+rangeControl("trainDropout","Dropout","p",0,.5,.05,0)+'<div class="toggle-row"><label><input id="trainEarly" type="checkbox"> Stop after 20 unhelpful epochs</label></div></div>';
  conceptElements.actions.innerHTML='<button class="button primary" id="trainAuto" type="button">▶ Train</button><button class="button" id="trainOne" type="button">Step 1 epoch</button><button class="button" id="trainTwenty" type="button">Run 10 epochs</button><button class="button" id="trainFreeze" type="button">Freeze model A</button><button class="button quiet" id="trainNew" type="button">New data</button><button class="button quiet" id="trainReset" type="button">Reset weights</button>';
  legend([{label:"Positive / class 1",color:conceptPalette.blue},{label:"Negative / class 0",color:conceptPalette.orange},{label:"Training loss",color:conceptPalette.green},{label:"Validation loss",color:conceptPalette.red}]);
  setExplanation(
    '<p>A hidden neuron turns a weighted mixture of earlier signals into a new pattern. Later neurons recombine those patterns. Training gradually adjusts every connection so the blue and orange examples receive different output probabilities.</p><div class="try-card"><strong>Try this</strong><p>Train the circle preset. Switch to one hidden layer with one neuron. Why can it not wrap a boundary around the centre? Add neurons one at a time and inspect their activation maps.</p></div><p class="concept-readout" id="trainingNodeReadout"><strong>Selected:</strong> output prediction map.</p>',
    '<div class="formula compact">a<sup>(ℓ)</sup> = activation(W<sup>(ℓ)</sup>a<sup>(ℓ−1)</sup> + b<sup>(ℓ)</sup>)</div><p id="trainingMath"></p>',
    '<ul class="plain-list"><li>Blue connections have positive weights; orange connections have negative weights. Thickness represents absolute magnitude.</li><li>Feature maps show a neuron’s activation over all possible x₁–x₂ locations—not a picture stored inside that neuron.</li><li>Each epoch visits every training example. Batch size controls how many examples contribute to one weight update.</li><li>Validation examples are never used for weight updates; outlined points show that held-out set.</li><li>L1 can drive weights toward zero, L2 discourages large weights smoothly, and dropout temporarily hides hidden activations during training.</li></ul>'
  );

  conceptState={
    view:"network",dataset:"circles",sampleCount:100,trainRatio:.5,noise:0,features:["x","y"],activation:"tanh",depth:2,width1:4,width2:2,
    rate:.03,batch:10,regularizer:"none",strength:0,dropout:0,early:false,seed:527,epoch:0,model:null,points:[],history:[],snapshot:null,
    bestLoss:Infinity,bestModel:null,bestEpoch:0,patience:0,stopped:false,selected:{kind:"output",index:0},hitAreas:[]
  };
  const settings=()=>({features:conceptState.features,activation:conceptState.activation,rate:conceptState.rate,regularizer:conceptState.regularizer,strength:conceptState.strength,dropout:conceptState.dropout});
  const widths=()=>conceptState.depth===1?[conceptState.width1]:[conceptState.width1,conceptState.width2];
  const split=()=>({train:conceptState.points.filter(point=>point.split==="train"),validation:conceptState.points.filter(point=>point.split==="validation")});
  const currentMetrics=(model=conceptState.model)=>{const groups=split(),options=settings();return {train:trainingEvaluate(model,groups.train,options),validation:trainingEvaluate(model,groups.validation,options)}};
  const record=()=>{const metrics=currentMetrics();conceptState.history.push({epoch:conceptState.epoch,train:metrics.train.loss,validation:metrics.validation.loss});if(conceptState.history.length>241)conceptState.history.shift();return metrics};
  const updateSecondLayer=()=>{const disabled=conceptState.depth===1,$section=$("#trainSecondLayer"),$input=$("#trainWidth2");if($section)$section.classList.toggle("disabled",disabled);if($input)$input.disabled=disabled};
  const rebuild=(newData=true)=>{
    stopConceptAnimation();
    if(newData)conceptState.points=trainingDataset(conceptState.dataset,conceptState.sampleCount,conceptState.noise,conceptState.trainRatio,conceptState.seed);
    conceptState.model=trainingCreateModel(conceptState.features.length,widths(),conceptState.seed+97);conceptState.epoch=0;conceptState.history=[];conceptState.snapshot=null;
    conceptState.bestLoss=Infinity;conceptState.bestModel=null;conceptState.bestEpoch=0;conceptState.patience=0;conceptState.stopped=false;conceptState.selected={kind:"output",index:0};
    const freezeButton=$("#trainFreeze");if(freezeButton)freezeButton.textContent="Freeze model A";
    const metrics=record();conceptState.bestLoss=metrics.validation.loss;conceptState.bestModel=trainingClone(conceptState.model);conceptRender();
  };
  const trainEpoch=(render=true)=>{
    if(conceptState.stopped)return;
    const train=[...split().train],random=seededRandom(conceptState.seed+conceptState.epoch*1009+31);
    for(let index=train.length-1;index>0;index--){const target=Math.floor(random()*(index+1));[train[index],train[target]]=[train[target],train[index]]}
    const batchSize=conceptState.batch==="all"?train.length:Number(conceptState.batch);
    for(let start=0;start<train.length;start+=batchSize)trainingGradientStep(conceptState.model,train.slice(start,start+batchSize),settings(),conceptState.seed+conceptState.epoch*2003+start);
    conceptState.epoch++;let metrics=record();
    if(metrics.validation.loss<conceptState.bestLoss-.0002){conceptState.bestLoss=metrics.validation.loss;conceptState.bestModel=trainingClone(conceptState.model);conceptState.bestEpoch=conceptState.epoch;conceptState.patience=0}else conceptState.patience++;
    if(conceptState.early&&conceptState.patience>=20){conceptState.model=trainingClone(conceptState.bestModel);conceptState.stopped=true;stopConceptAnimation();metrics=currentMetrics()}
    if(render)conceptRender();
  };
  const trainMany=count=>{stopConceptAnimation();for(let index=0;index<count&&!conceptState.stopped;index++)trainEpoch(false);conceptRender()};

  const heatColour=value=>{
    const amount=clamp(Math.abs(value),0,1),target=value>=0?[36,110,170]:[226,120,53],base=[247,249,250];
    const mix=.12+.78*amount;return `rgb(${target.map((channel,index)=>Math.round(base[index]*(1-mix)+channel*mix)).join(",")})`;
  };
  const nodeValue=(node,point)=>{
    if(node.kind==="input")return clamp(TRAINING_FEATURES[conceptState.features[node.index]].map(point),-1,1);
    const forward=trainingForward(conceptState.model,trainingFeatureVector(point,conceptState.features),conceptState.activation);
    if(node.kind==="hidden"){
      const value=forward.caches[node.layer].raw[node.index];
      return conceptState.activation==="relu"?clamp(value/2,0,1):clamp(value,-1,1);
    }
    return forward.probability*2-1;
  };
  const drawHeatTile=(context,node,box,points=false)=>{
    const cells=box.w<46?7:10,cellW=box.w/cells,cellH=box.h/cells;
    for(let row=0;row<cells;row++)for(let column=0;column<cells;column++){
      const point={x:-1.2+(column+.5)/cells*2.4,y:1.2-(row+.5)/cells*2.4};
      context.fillStyle=heatColour(nodeValue(node,point));context.fillRect(box.x+column*cellW,box.y+row*cellH,cellW+1,cellH+1);
    }
    if(points)conceptState.points.forEach(point=>{
      const x=box.x+(point.x+1.2)/2.4*box.w,y=box.y+(1.2-point.y)/2.4*box.h,r=point.split==="train"?Math.max(2.2,box.w/42):Math.max(1.9,box.w/48);
      context.beginPath();context.arc(x,y,r,0,Math.PI*2);
      if(point.split==="train"){context.fillStyle=point.label?conceptPalette.blue:conceptPalette.orange;context.fill();context.strokeStyle="#fff";context.lineWidth=.8;context.stroke()}
      else{context.fillStyle="#fff";context.fill();context.strokeStyle=point.label?conceptPalette.blue:conceptPalette.orange;context.lineWidth=1.4;context.stroke()}
    });
    const selected=conceptState.selected&&conceptState.selected.kind===node.kind&&conceptState.selected.index===node.index&&(node.kind!=="hidden"||conceptState.selected.layer===node.layer);
    context.strokeStyle=selected?conceptPalette.ink:"#8fa3b2";context.lineWidth=selected?3:1.2;context.strokeRect(box.x,box.y,box.w,box.h);
    conceptState.hitAreas.push({...box,node});
  };
  const nodePositions=(count,centerX,top,bottom,tile)=>Array.from({length:count},(_,index)=>({x:centerX-tile/2,y:count===1?(top+bottom-tile)/2:top+index*(bottom-top-tile)/(count-1),w:tile,h:tile}));
  const isSelectedEdge=(layer,from,to)=>{
    const selected=conceptState.selected;if(!selected)return false;
    if(selected.kind==="input"&&layer===0&&selected.index===from)return true;
    if(selected.kind==="hidden"&&((selected.layer===layer&&selected.index===to)||(selected.layer===layer-1&&selected.index===from)))return true;
    return selected.kind==="output"&&layer===conceptState.model.weights.length-1;
  };
  const drawLossMini=(context,box)=>{
    const values=conceptState.history;if(!values.length)return;const maxEpoch=Math.max(1,...values.map(value=>value.epoch)),maxLoss=Math.max(.75,...values.flatMap(value=>[value.train,value.validation]))*1.08;
    context.fillStyle=conceptPalette.muted;context.font="800 9px system-ui";context.textAlign="left";context.fillText("LOSS",box.x,box.y-6);context.strokeStyle="#cbd7df";context.lineWidth=1;context.strokeRect(box.x,box.y,box.w,box.h);
    const x=epoch=>box.x+epoch/maxEpoch*box.w,y=loss=>box.y+box.h-clamp(loss/maxLoss,0,1)*box.h;
    [["train",conceptPalette.green],["validation",conceptPalette.red]].forEach(([key,color])=>{context.beginPath();values.forEach((value,index)=>index?context.lineTo(x(value.epoch),y(value[key])):context.moveTo(x(value.epoch),y(value[key])));context.strokeStyle=color;context.lineWidth=2;context.stroke();const last=values[values.length-1];context.beginPath();context.arc(x(last.epoch),y(last[key]),2.7,0,Math.PI*2);context.fillStyle=color;context.fill()});
    context.fillStyle=conceptPalette.muted;context.font="9px system-ui";context.textAlign="right";context.fillText(`epoch ${conceptState.epoch}`,box.x+box.w,box.y+box.h+12);
  };
  const selectedText=metrics=>{
    const selected=conceptState.selected||{kind:"output",index:0};
    if(selected.kind==="input"){const feature=TRAINING_FEATURES[conceptState.features[selected.index]];return `<strong>Selected ${feature.label}:</strong> ${feature.long}. This prepared signal is sent to every first-layer neuron.`}
    if(selected.kind==="hidden"){
      const layer=selected.layer,node=selected.index,incoming=conceptState.model.weights[layer][node],largest=Math.max(...incoming.map(Math.abs)),source=incoming.findIndex(value=>Math.abs(value)===largest),sourceName=layer===0?TRAINING_FEATURES[conceptState.features[source]].label:`H${layer}:${source+1}`;
      return `<strong>Selected H${layer+1}:${node+1}:</strong> bias ${fmt(conceptState.model.biases[layer][node],2)}; strongest incoming link is ${sourceName} with weight ${fmt(incoming[source],2)}.`;
    }
    return `<strong>Selected output:</strong> blue means a higher class-1 probability. Validation accuracy is ${Math.round(metrics.validation.accuracy*100)}%.`;
  };
  const drawNetwork=(context,size,metrics)=>{
    const w=size.width,h=size.height,top=55,bottom=h-72,maxNodes=Math.max(conceptState.features.length,...widths()),available=bottom-top;
    const tile=clamp(Math.min(58,(available-(maxNodes-1)*7)/maxNodes),30,58),outSize=clamp(Math.min(122,w*.2,available*.38),68,122),inputX=18+tile/2,outX=w-18-outSize/2;
    const hiddenXs=conceptState.depth===2?[inputX+(outX-inputX)*.34,inputX+(outX-inputX)*.67]:[inputX+(outX-inputX)*.49];
    const inputBoxes=nodePositions(conceptState.features.length,inputX,top,bottom,tile),hiddenBoxes=widths().map((count,layer)=>nodePositions(count,hiddenXs[layer],top,bottom,tile));
    const outputBox={x:outX-outSize/2,y:top+available*.34-outSize/2,w:outSize,h:outSize};
    const columns=[inputBoxes,...hiddenBoxes,[outputBox]];conceptState.hitAreas=[];
    context.fillStyle=conceptPalette.muted;context.font="800 10px system-ui";context.textAlign="center";
    context.fillText("FEATURES",inputX,24);hiddenXs.forEach((x,index)=>context.fillText(`HIDDEN ${index+1}`,x,24));context.fillText("OUTPUT",outX,24);
    context.font="9px system-ui";context.fillStyle="#7a8a98";context.fillText(`${conceptState.activation} activation`,hiddenXs[Math.floor((hiddenXs.length-1)/2)],39);
    conceptState.model.weights.forEach((weights,layer)=>weights.forEach((row,to)=>row.forEach((weight,from)=>{
      const source=columns[layer][from],target=columns[layer+1][to];if(!source||!target)return;const highlighted=isSelectedEdge(layer,from,to);
      context.beginPath();context.moveTo(source.x+source.w,source.y+source.h/2);context.bezierCurveTo((source.x+source.w+target.x)/2,source.y+source.h/2,(source.x+source.w+target.x)/2,target.y+target.h/2,target.x,target.y+target.h/2);
      context.strokeStyle=(weight>=0?conceptPalette.blue:conceptPalette.orange)+(highlighted?"e8":"76");context.lineWidth=clamp(.55+Math.abs(weight)*1.5,.55,5);context.stroke();
    })));
    inputBoxes.forEach((box,index)=>{drawHeatTile(context,{kind:"input",index},box);context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign="center";context.fillText(TRAINING_FEATURES[conceptState.features[index]].label,box.x+box.w/2,box.y+box.h+12)});
    hiddenBoxes.forEach((boxes,layer)=>boxes.forEach((box,index)=>{drawHeatTile(context,{kind:"hidden",layer,index},box);context.fillStyle=conceptPalette.ink;context.font="700 9px system-ui";context.textAlign="center";context.fillText(`H${layer+1}:${index+1}`,box.x+box.w/2,box.y+box.h+11)}));
    drawHeatTile(context,{kind:"output",index:0},outputBox,true);
    context.fillStyle=conceptPalette.muted;context.font="700 9px system-ui";context.textAlign="center";context.fillText("outlined = validation",outX,outputBox.y+outputBox.h+13);
    const chartWidth=Math.min(154,outSize*1.35),chartY=outputBox.y+outputBox.h+39,chartHeight=Math.max(35,Math.min(60,h-chartY-25));drawLossMini(context,{x:outX-chartWidth/2,y:chartY,w:chartWidth,h:chartHeight});
    context.fillStyle="#edf3f6";context.fillRect(0,h-33,w,33);context.fillStyle=conceptPalette.muted;context.font="700 10px system-ui";context.textAlign="left";context.fillText("CLICK A MAP TO INSPECT IT",12,h-12);
  };
  const drawDecision=(context,model,box,title,subtitle)=>{
    const plot={x:box.x,y:box.y+29,w:box.w,h:box.h-29},cells=Math.max(15,Math.min(28,Math.round(plot.w/9))),cellW=plot.w/cells,cellH=plot.h/cells;
    context.fillStyle=conceptPalette.ink;context.font="800 11px system-ui";context.textAlign="left";context.fillText(title,box.x,box.y+10);context.fillStyle=conceptPalette.muted;context.font="10px system-ui";context.fillText(subtitle,box.x,box.y+23);
    for(let row=0;row<cells;row++)for(let column=0;column<cells;column++){
      const point={x:-1.2+(column+.5)/cells*2.4,y:1.2-(row+.5)/cells*2.4},probability=trainingForward(model,trainingFeatureVector(point,conceptState.features),conceptState.activation).probability;
      context.fillStyle=heatColour(probability*2-1);context.fillRect(plot.x+column*cellW,plot.y+row*cellH,cellW+1,cellH+1);
    }
    context.strokeStyle="#8ea2b2";context.lineWidth=1.2;context.strokeRect(plot.x,plot.y,plot.w,plot.h);
    conceptState.points.forEach(point=>{const x=plot.x+(point.x+1.2)/2.4*plot.w,y=plot.y+(1.2-point.y)/2.4*plot.h;context.beginPath();context.arc(x,y,point.split==="train"?3.5:2.8,0,Math.PI*2);if(point.split==="train"){context.fillStyle=point.label?conceptPalette.blue:conceptPalette.orange;context.fill();context.strokeStyle="#fff";context.lineWidth=1;context.stroke()}else{context.fillStyle="#fff";context.fill();context.strokeStyle=point.label?conceptPalette.blue:conceptPalette.orange;context.lineWidth=1.7;context.stroke()}});
  };
  const drawCurves=(context,box)=>{
    const values=conceptState.history,maxEpoch=Math.max(1,...values.map(value=>value.epoch)),maxLoss=Math.max(.75,...values.flatMap(value=>[value.train,value.validation]))*1.08,chart={x:box.x+37,y:box.y+31,w:box.w-49,h:box.h-57};
    context.fillStyle=conceptPalette.ink;context.font="800 11px system-ui";context.textAlign="left";context.fillText("LOSS ACROSS EPOCHS",box.x,box.y+11);context.strokeStyle="#dce5eb";context.lineWidth=1;
    for(let index=0;index<=4;index++){const y=chart.y+index/4*chart.h;context.beginPath();context.moveTo(chart.x,y);context.lineTo(chart.x+chart.w,y);context.stroke();context.fillStyle=conceptPalette.muted;context.font="9px system-ui";context.textAlign="right";context.fillText(fmt(maxLoss*(1-index/4),2),chart.x-5,y+3)}
    context.strokeStyle="#91a2b1";context.strokeRect(chart.x,chart.y,chart.w,chart.h);const x=epoch=>chart.x+epoch/maxEpoch*chart.w,y=loss=>chart.y+chart.h-clamp(loss/maxLoss,0,1)*chart.h;
    [["train",conceptPalette.green],["validation",conceptPalette.red]].forEach(([key,color])=>{context.beginPath();values.forEach((value,index)=>index?context.lineTo(x(value.epoch),y(value[key])):context.moveTo(x(value.epoch),y(value[key])));context.strokeStyle=color;context.lineWidth=2.5;context.stroke();const last=values[values.length-1];context.beginPath();context.arc(x(last.epoch),y(last[key]),3.5,0,Math.PI*2);context.fillStyle=color;context.fill()});
    if(conceptState.snapshot){const marker=x(conceptState.snapshot.epoch);context.save();context.setLineDash([4,4]);context.beginPath();context.moveTo(marker,chart.y);context.lineTo(marker,chart.y+chart.h);context.strokeStyle=conceptPalette.purple;context.lineWidth=2;context.stroke();context.restore();context.fillStyle=conceptPalette.purple;context.font="800 10px system-ui";context.textAlign="center";context.fillText("A",marker,chart.y-6)}
    context.fillStyle=conceptPalette.muted;context.font="10px system-ui";context.textAlign="right";context.fillText(`epoch ${maxEpoch}`,chart.x+chart.w,chart.y+chart.h+17);context.textAlign="left";context.fillText("0",chart.x,chart.y+chart.h+17);
  };
  const drawComparison=(context,size)=>{
    const w=size.width,h=size.height,mobile=w<620;
    if(mobile){if(conceptState.snapshot){const gap=10,mapWidth=(w-44-gap)/2;drawDecision(context,conceptState.snapshot.model,{x:22,y:18,w:mapWidth,h:205},"MODEL A",`epoch ${conceptState.snapshot.epoch}`);drawDecision(context,conceptState.model,{x:22+mapWidth+gap,y:18,w:mapWidth,h:205},"MODEL B",`epoch ${conceptState.epoch}`)}else drawDecision(context,conceptState.model,{x:Math.max(22,(w-Math.min(300,w-44))/2),y:18,w:Math.min(300,w-44),h:205},"CURRENT MODEL",`epoch ${conceptState.epoch}`);drawCurves(context,{x:18,y:245,w:w-36,h:h-255});return}
    if(conceptState.snapshot){const pad=22,gap=14,mapWidth=Math.min(225,(w*.61-pad*2-gap)/2),chartX=pad+mapWidth*2+gap*2;drawDecision(context,conceptState.snapshot.model,{x:pad,y:42,w:mapWidth,h:h-78},"MODEL A",`frozen at epoch ${conceptState.snapshot.epoch}`);drawDecision(context,conceptState.model,{x:pad+mapWidth+gap,y:42,w:mapWidth,h:h-78},"MODEL B",`current epoch ${conceptState.epoch}`);drawCurves(context,{x:chartX,y:42,w:w-chartX-pad,h:h-78})}
    else{const mapWidth=Math.min(360,w*.49);drawDecision(context,conceptState.model,{x:24,y:42,w:mapWidth,h:h-78},"CURRENT DECISION MAP",`epoch ${conceptState.epoch}`);drawCurves(context,{x:mapWidth+52,y:42,w:w-mapWidth-76,h:h-78})}
  };

  conceptRender=()=>{
    const size=conceptSize(),context=conceptContext,metrics=currentMetrics(),snapshotMetrics=conceptState.snapshot?conceptState.snapshot.metrics:null;context.clearRect(0,0,size.width,size.height);context.fillStyle="#f8fbfd";context.fillRect(0,0,size.width,size.height);
    if(conceptState.view==="network")drawNetwork(context,size,metrics);else drawComparison(context,size);
    const gap=metrics.validation.loss-metrics.train.loss,comparison=snapshotMetrics?metrics.validation.loss-snapshotMetrics.validation.loss:null;
    metricCards([
      {label:"Epoch",value:conceptState.epoch,note:conceptState.stopped?`Restored epoch ${conceptState.bestEpoch}`:`Best validation at ${conceptState.bestEpoch}`},
      {label:"Training loss",value:fmt(metrics.train.loss,4),note:`${split().train.length} filled examples`},
      {label:"Validation loss",value:fmt(metrics.validation.loss,4),note:`Gap ${gap>=0?"+":""}${fmt(gap,4)}`},
      {label:snapshotMetrics?"A → B validation Δ":"Validation accuracy",value:snapshotMetrics?`${comparison>=0?"+":""}${fmt(comparison,4)}`:`${Math.round(metrics.validation.accuracy*100)}%`,note:snapshotMetrics?(comparison<0?"B generalizes better":"A generalizes better"):`${split().validation.length} outlined examples`}
    ]);
    const math=$("#trainingMath"),readout=$("#trainingNodeReadout"),architecture=[conceptState.features.length,...widths(),1].join(" → "),regularization=conceptState.regularizer==="none"?"no weight penalty":`${conceptState.regularizer.toUpperCase()} λ = ${fmt(conceptState.strength,3)}`;
    if(math)math.innerHTML=`Architecture <b>${architecture}</b> uses <b>${conceptState.activation}</b> hidden activations. Epoch ${conceptState.epoch}: training loss <b>${fmt(metrics.train.loss,4)}</b>, validation loss <b>${fmt(metrics.validation.loss,4)}</b>, ${regularization}.`;
    if(readout)readout.innerHTML=selectedText(metrics);
  };

  document.querySelectorAll('input[name="trainView"]').forEach(input=>input.addEventListener("change",event=>{if(!event.target.checked)return;conceptState.view=event.target.value;conceptRender()}));
  document.querySelectorAll('input[name="trainDataset"]').forEach(input=>input.addEventListener("change",event=>{if(!event.target.checked)return;conceptState.dataset=event.target.value;rebuild(true)}));
  bindRange("trainSamples",value=>{conceptState.sampleCount=value;rebuild(true)},0);bindRange("trainRatio",value=>{conceptState.trainRatio=value/100;rebuild(true)},0);bindRange("trainNoise",value=>{conceptState.noise=value;rebuild(true)},2);
  const featureInputs=["trainFeatureX","trainFeatureY","trainFeatureXY","trainFeatureX2","trainFeatureY2"];
  featureInputs.forEach(id=>$("#"+id).addEventListener("change",event=>{let chosen=featureInputs.map(inputId=>$("#"+inputId)).filter(input=>input.checked).map(input=>input.value);if(!chosen.length&&event.isTrusted){event.target.checked=true;chosen=[event.target.value]}conceptState.features=chosen.length?chosen:["x"];rebuild(false)}));
  $("#trainActivation").value=conceptState.activation;$("#trainActivation").addEventListener("change",event=>{conceptState.activation=event.target.value;rebuild(false)});
  $("#trainDepth").value=String(conceptState.depth);$("#trainDepth").addEventListener("change",event=>{conceptState.depth=Number(event.target.value);updateSecondLayer();rebuild(false)});
  bindRange("trainWidth1",value=>{conceptState.width1=value;rebuild(false)},0);bindRange("trainWidth2",value=>{conceptState.width2=value;rebuild(false)},0);
  $("#trainRate").value=String(conceptState.rate);$("#trainRate").addEventListener("change",event=>conceptState.rate=Number(event.target.value));
  $("#trainBatch").value=String(conceptState.batch);$("#trainBatch").addEventListener("change",event=>conceptState.batch=event.target.value==="all"?"all":Number(event.target.value));
  $("#trainRegularizer").value=conceptState.regularizer;$("#trainRegularizer").addEventListener("change",event=>{conceptState.regularizer=event.target.value;conceptState.stopped=false;conceptRender()});
  bindRange("trainStrength",value=>{conceptState.strength=value;conceptState.stopped=false},3);bindRange("trainDropout",value=>{conceptState.dropout=value;conceptState.stopped=false},2);
  $("#trainEarly").addEventListener("change",event=>{conceptState.early=event.target.checked;conceptState.stopped=false;conceptRender()});
  $("#trainOne").addEventListener("click",()=>{stopConceptAnimation();trainEpoch()});$("#trainTwenty").addEventListener("click",()=>trainMany(10));
  $("#trainAuto").addEventListener("click",()=>{if(conceptTimer){stopConceptAnimation();return}conceptState.stopped=false;$("#trainAuto").textContent="Pause";conceptTimer=setInterval(()=>trainEpoch(),window.labDelay?window.labDelay(150):150)});
  $("#trainFreeze").addEventListener("click",()=>{const metrics=currentMetrics();conceptState.snapshot={model:trainingClone(conceptState.model),metrics,epoch:conceptState.epoch};$("#trainFreeze").textContent="Replace model A";conceptState.view="curves";$("#trainViewCurves").checked=true;conceptRender()});
  $("#trainNew").addEventListener("click",()=>{conceptState.seed++;rebuild(true)});$("#trainReset").addEventListener("click",()=>rebuild(false));
  conceptPointerDown=event=>{if(conceptState.view!=="network")return;const position=conceptPosition(event),hit=conceptState.hitAreas.find(area=>position.x>=area.x&&position.x<=area.x+area.w&&position.y>=area.y&&position.y<=area.y+area.h);if(hit){conceptState.selected={...hit.node};conceptRender()}};
  conceptCanvas.style.cursor="pointer";conceptCanvas.setAttribute("aria-label","Interactive neural network. Feature and neuron squares are activation maps; click a square to inspect it. Use the controls and training buttons to change the network.");updateSecondLayer();rebuild(true);
}

CONCEPT_SETUPS.training=setupTraining;
