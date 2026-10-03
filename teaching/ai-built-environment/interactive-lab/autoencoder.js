"use strict";

CONCEPT_META.autoencoder={
  category:"Deep Learning",
  title:"Autoencoder & Latent Space Playground",
  summary:"Draw an input, compress it through a bottleneck, reconstruct it and explore how nearby latent codes create related outputs.",
  question:"What information survives compression, and what does distance mean in a latent space?"
};

const AE_PLAY_ANCHORS={
  t:{label:"T",x:-.78,y:.76},
  x:{label:"X",x:.78,y:.76},
  box:{label:"Box",x:-.78,y:-.76},
  bars:{label:"Bars",x:.78,y:-.76}
};

function aePlayPattern(name){
  const size=8,grid=Array.from({length:size},()=>Array(size).fill(0));
  for(let row=0;row<size;row++)for(let column=0;column<size;column++){
    if(name==="x")grid[row][column]=row===column||row+column===size-1?1:0;
    else if(name==="box")grid[row][column]=row===1||row===6||column===1||column===6?1:0;
    else if(name==="bars")grid[row][column]=column===1||column===3||column===6?1:0;
    else grid[row][column]=(row===1&&column>=1&&column<=6)||(column===3&&row>=1&&row<=6)?1:0;
  }
  return grid;
}
function aePlayClone(matrix){return matrix.map(row=>[...row])}
function aePlayMse(first,second){return first.flat().reduce((sum,value,index)=>sum+(value-second.flat()[index])**2,0)/first.flat().length}
function aePlayDctAlpha(index,size){return index===0?Math.sqrt(1/size):Math.sqrt(2/size)}
let AE_PLAY_DCT_ORDER=null;
function aePlayDctOrder(size){
  if(AE_PLAY_DCT_ORDER)return AE_PLAY_DCT_ORDER;
  const positions=Array.from({length:size*size},(_,index)=>({u:Math.floor(index/size),v:index%size,energy:0}));
  Object.keys(AE_PLAY_ANCHORS).forEach(name=>{const coefficients=aePlayDctEncode(aePlayPattern(name));positions.forEach(position=>{position.energy+=coefficients[position.u][position.v]**2})});
  AE_PLAY_DCT_ORDER=positions.sort((a,b)=>b.energy-a.energy||(a.u+a.v)-(b.u+b.v)).map(({u,v})=>({u,v}));
  return AE_PLAY_DCT_ORDER;
}
function aePlayDctEncode(matrix){
  const size=matrix.length,coefficients=Array.from({length:size},()=>Array(size).fill(0));
  for(let u=0;u<size;u++)for(let v=0;v<size;v++){
    let sum=0;
    for(let row=0;row<size;row++)for(let column=0;column<size;column++)sum+=matrix[row][column]*Math.cos((2*row+1)*u*Math.PI/(2*size))*Math.cos((2*column+1)*v*Math.PI/(2*size));
    coefficients[u][v]=aePlayDctAlpha(u,size)*aePlayDctAlpha(v,size)*sum;
  }
  return coefficients;
}
function aePlayDctDecode(coefficients,keep){
  const size=coefficients.length,output=Array.from({length:size},()=>Array(size).fill(0)),positions=aePlayDctOrder(size).slice(0,keep);
  for(let row=0;row<size;row++)for(let column=0;column<size;column++){
    let sum=0;
    positions.forEach(({u,v})=>{sum+=aePlayDctAlpha(u,size)*aePlayDctAlpha(v,size)*coefficients[u][v]*Math.cos((2*row+1)*u*Math.PI/(2*size))*Math.cos((2*column+1)*v*Math.PI/(2*size))});
    output[row][column]=clamp(sum,0,1);
  }
  return output;
}
function aePlayCorrupt(matrix,noise,seed){
  const random=seededRandom(seed);
  return matrix.map(row=>row.map(value=>clamp(value+(random()-.5)*2*noise,0,1)));
}
function aePlaySmooth(matrix,strength){
  if(strength<=0)return aePlayClone(matrix);
  const size=matrix.length;
  return matrix.map((row,rowIndex)=>row.map((value,columnIndex)=>{
    let total=0,weight=0;
    for(let rowOffset=-1;rowOffset<=1;rowOffset++)for(let columnOffset=-1;columnOffset<=1;columnOffset++){
      const sourceRow=clamp(rowIndex+rowOffset,0,size-1),sourceColumn=clamp(columnIndex+columnOffset,0,size-1),itemWeight=rowOffset===0&&columnOffset===0?4:(Math.abs(rowOffset)+Math.abs(columnOffset)===1?2:1);
      total+=matrix[sourceRow][sourceColumn]*itemWeight;weight+=itemWeight;
    }
    return value*(1-strength)+(total/weight)*strength;
  }));
}
function aePlayReconstruct(observed,features,task,noise){
  const prepared=task==="denoise"?aePlaySmooth(observed,clamp(noise*.35,0,.32)):observed,coefficients=aePlayDctEncode(prepared),order=aePlayDctOrder(8),kept=order.slice(0,features),reconstruction=aePlayDctDecode(coefficients,features),totalEnergy=coefficients.flat().reduce((sum,value)=>sum+value*value,0),keptEnergy=kept.reduce((sum,{u,v})=>sum+coefficients[u][v]**2,0);
  return {prepared,coefficients,kept,reconstruction,energy:keptEnergy/(totalEnergy||1)};
}
function aePlayLatentWeights(x,y){
  const entries=Object.entries(AE_PLAY_ANCHORS),raw=entries.map(([key,anchor])=>({key,value:Math.exp(-((x-anchor.x)**2+(y-anchor.y)**2)/.32)})),total=raw.reduce((sum,item)=>sum+item.value,0)||1;
  return raw.map(item=>({...item,value:item.value/total}));
}
function aePlayLatentDecode(x,y){
  const weights=aePlayLatentWeights(x,y),output=Array.from({length:8},()=>Array(8).fill(0));
  weights.forEach(({key,value})=>aePlayPattern(key).forEach((row,rowIndex)=>row.forEach((pixel,columnIndex)=>{output[rowIndex][columnIndex]+=pixel*value})));
  return {output,weights};
}
function aePlayLatentEncode(matrix){
  const raw=Object.entries(AE_PLAY_ANCHORS).map(([key,anchor])=>({key,anchor,value:Math.exp(-aePlayMse(matrix,aePlayPattern(key))*9)})),total=raw.reduce((sum,item)=>sum+item.value,0)||1;
  return {x:raw.reduce((sum,item)=>sum+item.anchor.x*item.value,0)/total,y:raw.reduce((sum,item)=>sum+item.anchor.y*item.value,0)/total};
}

function setupAutoencoderPlayground(){
  setConcept(
    CONCEPT_META.autoencoder,
    "Representation playground",
    "Compress, reconstruct and navigate a latent space",
    "In reconstruction view, click the clean input pixels to draw. In latent-space view, drag the purple code or use the arrow keys to generate nearby outputs.",
    "<strong>Start here</strong><p>Add noise to the diagonal X, choose <b>Denoise toward clean target</b>, then reduce the bottleneck. Find the smallest code that preserves the shape.</p>"
  );
  conceptElements.controls.innerHTML=
    '<span class="field-label">Visualization</span><div class="train-view-switch ae-view-switch" role="radiogroup" aria-label="Autoencoder visualization"><label><input id="aeViewPipeline" type="radio" name="aeView" value="pipeline" checked><span>Reconstruction</span></label><label><input id="aeViewLatent" type="radio" name="aeView" value="latent"><span>Latent space</span></label></div>'+
    '<div class="train-control-section"><span class="train-control-heading">1 · Input</span>'+selectControl("aePattern","Input pattern",[["t","Letter T"],["x","Diagonal X"],["box","Box"],["bars","Vertical bars"],["custom","Custom drawing"]])+rangeControl("aeNoise","Input noise","η",0,.8,.05,.25)+'</div>'+
    '<div class="train-control-section" id="aePipelineControls"><span class="train-control-heading">2 · Compression</span>'+rangeControl("aeCode","Bottleneck features","K",1,16,1,6)+selectControl("aeTask","Learning task",[["reconstruct","Reconstruct model input"],["denoise","Denoise toward clean target"]])+'</div>'+
    '<div class="train-control-section" id="aeLatentControls"><span class="train-control-heading">3 · Latent explorer</span>'+rangeControl("aeLatentX","Horizontal code","z₁",-1,1,.05,-.78)+rangeControl("aeLatentY","Vertical code","z₂",-1,1,.05,.76)+'</div>';
  conceptElements.actions.innerHTML='<button class="button primary" id="aeNoiseNew" type="button">New noise</button><button class="button" id="aeUseEncoded" type="button">Use encoded point</button><button class="button quiet" id="aeClear" type="button">Clear drawing</button><button class="button quiet" id="aeReset" type="button">Reset</button>';
  setExplanation(
    '<p>An autoencoder sends an input through an encoder, represents it with a smaller latent code, and uses a decoder to construct an output. Compression forces the code to preserve some structure and discard other detail.</p><div class="try-card"><strong>Try this</strong><p>Compare <b>Reconstruct model input</b> with <b>Denoise toward clean target</b>. The same noisy input can be paired with a different learning objective.</p></div><p class="concept-readout" id="aeReadout"><strong>Bottleneck:</strong> six values replace the original 64 pixels.</p>',
    '<div class="formula compact" id="aeFormula">z = encoder(x) &nbsp; · &nbsp; x̂ = decoder(z)</div><p id="aeMath"></p>',
    '<ul class="plain-list"><li>The bottleneck limits how much information can pass through the model.</li><li>A denoising autoencoder receives a corrupted input but learns against a clean target.</li><li>Nearby latent codes often decode to similar outputs, making interpolation possible.</li><li>A latent coordinate is not automatically a human-readable concept; its meaning comes from the learned representation.</li><li>This transparent playground uses prepared frequency features and a prepared 2D map. A neural autoencoder learns both mappings from data.</li></ul>'
  );

  conceptState={view:"pipeline",pattern:"t",lastPattern:"t",clean:aePlayPattern("t"),features:6,noise:.25,task:"reconstruct",seed:881,latentX:-.78,latentY:.76,inputArea:null,latentArea:null,dragging:false};
  const observed=()=>aePlayCorrupt(conceptState.clean,conceptState.noise,conceptState.seed);
  const syncLatentControls=()=>{
    const x=$("#aeLatentX"),y=$("#aeLatentY");x.value=conceptState.latentX;y.value=conceptState.latentY;$("#aeLatentXOut").textContent=fmt(conceptState.latentX,2);$("#aeLatentYOut").textContent=fmt(conceptState.latentY,2);
  };
  const updateControlState=()=>{
    const pipeline=conceptState.view==="pipeline",pipelineGroup=$("#aePipelineControls"),latentGroup=$("#aeLatentControls");pipelineGroup.classList.toggle("disabled",!pipeline);latentGroup.classList.toggle("disabled",pipeline);pipelineGroup.querySelectorAll("input,select").forEach(control=>control.disabled=!pipeline);latentGroup.querySelectorAll("input,select").forEach(control=>control.disabled=pipeline);
  };
  const updateLegend=()=>conceptState.view==="pipeline"?legend([{label:"Clean target",color:conceptPalette.blue},{label:"Retained code",color:conceptPalette.orange},{label:"Reconstruction",color:conceptPalette.purple}]):legend([{label:"Pattern anchors",color:conceptPalette.green},{label:"Encoded input",color:conceptPalette.blue},{label:"Draggable code",color:conceptPalette.purple}]);
  const imageMatrix=(context,matrix,x,y,cell,label,editable=false)=>{
    context.fillStyle=conceptPalette.ink;context.font=`800 ${cell<12?9:10}px system-ui`;context.textAlign="left";context.fillText(label,x,y-10);
    matrix.forEach((row,rowIndex)=>row.forEach((value,columnIndex)=>{const amount=clamp(value,0,1),red=Math.round(247-amount*143),green=Math.round(250-amount*112),blue=Math.round(252-amount*72);context.fillStyle=`rgb(${red},${green},${blue})`;context.fillRect(x+columnIndex*cell,y+rowIndex*cell,cell,cell);context.strokeStyle="#c8d4dd";context.lineWidth=.8;context.strokeRect(x+columnIndex*cell,y+rowIndex*cell,cell,cell)}));
    context.strokeStyle=editable?conceptPalette.ink:"#aebdc8";context.lineWidth=editable?2:1.4;context.strokeRect(x,y,cell*8,cell*8);
  };
  const drawTradeoff=(context,chart,errors,current)=>{
    const maximum=Math.max(.02,...errors)*1.12,x=index=>chart.x+index/(errors.length-1)*chart.w,y=value=>chart.y+chart.h-value/maximum*chart.h;
    context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign="left";context.fillText("RECONSTRUCTION ERROR VS BOTTLENECK WIDTH",chart.x,chart.y-12);context.strokeStyle="#dce5eb";for(let line=0;line<=3;line++){const py=chart.y+line/3*chart.h;context.beginPath();context.moveTo(chart.x,py);context.lineTo(chart.x+chart.w,py);context.stroke()}context.strokeStyle="#91a4b2";context.strokeRect(chart.x,chart.y,chart.w,chart.h);context.beginPath();errors.forEach((value,index)=>index?context.lineTo(x(index),y(value)):context.moveTo(x(index),y(value)));context.strokeStyle=conceptPalette.purple;context.lineWidth=2.8;context.stroke();errors.forEach((value,index)=>{context.beginPath();context.arc(x(index),y(value),index===current?5:2.3,0,Math.PI*2);context.fillStyle=index===current?conceptPalette.orange:conceptPalette.purple;context.fill()});context.fillStyle=conceptPalette.muted;context.font="700 9px system-ui";context.textAlign="center";[0,3,7,11,15].forEach(index=>context.fillText(String(index+1),x(index),chart.y+chart.h+14));context.textAlign="right";context.fillText("more retained features →",chart.x+chart.w,chart.y+chart.h+28);
  };
  const pipelineRender=(context,size,input,result,target)=>{
    const mobile=size.width<560,margin=mobile?10:22,cell=mobile?Math.min(11.5,(size.width-60)/24):Math.min(17,(size.width-150)/24),matrixWidth=cell*8,y=58,cleanX=margin,inputX=(size.width-matrixWidth)/2,reconX=size.width-margin-matrixWidth;
    imageMatrix(context,conceptState.clean,cleanX,y,cell,mobile?(conceptState.task==="denoise"?"TARGET":"REFERENCE"):(conceptState.task==="denoise"?"CLEAN TARGET":"CLEAN REFERENCE"));imageMatrix(context,input,inputX,y,cell,mobile?(conceptState.noise?"NOISY INPUT":"MODEL INPUT"):(conceptState.noise?"MODEL INPUT + NOISE":"MODEL INPUT"),true);imageMatrix(context,result.reconstruction,reconX,y,cell,mobile?"OUTPUT":"OUTPUT x̂");conceptState.inputArea={x:inputX,y,cell,size:8};conceptState.latentArea=null;
    const flowY=y+matrixWidth+26,bottleX=mobile?34:70,bottleW=size.width-bottleX*2,slotGap=3,slotW=(bottleW-slotGap*15)/16,maxMagnitude=Math.max(.001,...result.kept.map(({u,v})=>Math.abs(result.coefficients[u][v])));context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign="center";context.fillText(`ENCODER  →  BOTTLENECK z · ${conceptState.features} / 64  →  DECODER`,size.width/2,flowY-12);for(let index=0;index<16;index++){const x=bottleX+index*(slotW+slotGap),position=aePlayDctOrder(8)[index],value=result.coefficients[position.u][position.v],active=index<conceptState.features,height=active?8+18*Math.abs(value)/maxMagnitude:8;context.fillStyle=active?(value>=0?conceptPalette.blue:conceptPalette.orange):"#e2e9ee";context.globalAlpha=active?1:.7;context.fillRect(x,flowY+24-height,slotW,height);context.globalAlpha=1;context.strokeStyle=index===conceptState.features-1?conceptPalette.ink:"#c2ced7";context.lineWidth=index===conceptState.features-1?2:1;context.strokeRect(x,flowY,slotW,24)}
    const errors=Array.from({length:16},(_,index)=>aePlayMse(aePlayReconstruct(input,index+1,conceptState.task,conceptState.noise).reconstruction,target)),chartY=mobile?315:334;drawTradeoff(context,{x:mobile?36:58,y:chartY,w:size.width-(mobile?54:82),h:mobile?85:92},errors,conceptState.features-1);
  };
  const latentRender=(context,size,input)=>{
    const mobile=size.width<560,plane=mobile?{x:44,y:54,w:size.width-68,h:225}:{x:55,y:55,w:size.width*.54,h:335},px=value=>plane.x+(value+1)/2*plane.w,py=value=>plane.y+(1-value)/2*plane.h,decoded=aePlayLatentDecode(conceptState.latentX,conceptState.latentY),encoded=aePlayLatentEncode(input);conceptState.latentArea=plane;conceptState.inputArea=null;
    context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign="left";context.fillText("PREPARED 2D LATENT MAP",plane.x,plane.y-15);context.fillStyle="#f5f8fa";context.fillRect(plane.x,plane.y,plane.w,plane.h);context.strokeStyle="#dce5eb";for(let line=0;line<=4;line++){const x=plane.x+line/4*plane.w,y=plane.y+line/4*plane.h;context.beginPath();context.moveTo(x,plane.y);context.lineTo(x,plane.y+plane.h);context.stroke();context.beginPath();context.moveTo(plane.x,y);context.lineTo(plane.x+plane.w,y);context.stroke()}context.strokeStyle="#91a4b2";context.strokeRect(plane.x,plane.y,plane.w,plane.h);context.beginPath();context.moveTo(px(0),plane.y);context.lineTo(px(0),plane.y+plane.h);context.moveTo(plane.x,py(0));context.lineTo(plane.x+plane.w,py(0));context.strokeStyle="#b2c0ca";context.stroke();
    Object.entries(AE_PLAY_ANCHORS).forEach(([key,anchor],anchorIndex)=>{const jitter=[[-.07,.04],[.05,.06],[-.03,-.07]];jitter.forEach(offset=>{context.beginPath();context.arc(px(clamp(anchor.x+offset[0],-1,1)),py(clamp(anchor.y+offset[1],-1,1)),3.2,0,Math.PI*2);context.fillStyle=conceptPalette.green;context.globalAlpha=.35;context.fill();context.globalAlpha=1});context.beginPath();context.arc(px(anchor.x),py(anchor.y),8,0,Math.PI*2);context.fillStyle=conceptPalette.green;context.fill();context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign=anchor.x<0?"left":"right";context.fillText(anchor.label,px(anchor.x)+(anchor.x<0?12:-12),py(anchor.y)+4)});
    context.beginPath();context.arc(px(encoded.x),py(encoded.y),9,0,Math.PI*2);context.strokeStyle=conceptPalette.blue;context.lineWidth=3;context.stroke();context.beginPath();context.arc(px(conceptState.latentX),py(conceptState.latentY),10,0,Math.PI*2);context.fillStyle=conceptPalette.purple;context.fill();context.strokeStyle="#fff";context.lineWidth=2;context.stroke();context.fillStyle=conceptPalette.muted;context.font="700 10px system-ui";context.textAlign="right";context.fillText("z₁",plane.x+plane.w,plane.y+plane.h+20);context.textAlign="left";context.fillText("z₂",plane.x-27,plane.y+10);
    if(mobile){const cell=11,x=(size.width-cell*8)/2,y=328;imageMatrix(context,decoded.output,x,y,cell,"DECODED OUTPUT");context.fillStyle=conceptPalette.muted;context.font="700 9px system-ui";context.textAlign="center";const strongest=[...decoded.weights].sort((a,b)=>b.value-a.value)[0];context.fillText(`${AE_PLAY_ANCHORS[strongest.key].label} contributes ${Math.round(strongest.value*100)}%`,size.width/2,y+cell*8+18)}
    else{const outputX=plane.x+plane.w+48,cell=Math.min(20,(size.width-outputX-20)/8),outputY=95;imageMatrix(context,decoded.output,outputX,outputY,cell,"DECODED OUTPUT");context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign="left";context.fillText("LOCAL MIX",outputX,outputY+cell*8+34);decoded.weights.sort((a,b)=>b.value-a.value).forEach((item,index)=>{const y=outputY+cell*8+52+index*28;context.fillStyle="#e2e9ee";context.fillRect(outputX,y,cell*8,11);context.fillStyle=index===0?conceptPalette.purple:conceptPalette.green;context.fillRect(outputX,y,cell*8*item.value,11);context.fillStyle=conceptPalette.muted;context.font="700 9px system-ui";context.fillText(`${AE_PLAY_ANCHORS[item.key].label} ${Math.round(item.value*100)}%`,outputX,y-4)})}
    return {decoded,encoded};
  };
  const updateExplanation=(input,result,target,latentData)=>{
    const math=$("#aeMath"),formula=$("#aeFormula"),readout=$("#aeReadout");
    if(conceptState.view==="pipeline"){
      const error=aePlayMse(result.reconstruction,target),cleanError=aePlayMse(result.reconstruction,conceptState.clean),noiseError=aePlayMse(input,conceptState.clean),gain=noiseError?100*(noiseError-cleanError)/noiseError:0;if(formula)formula.innerHTML=conceptState.task==="denoise"?"z = encoder(x̃) &nbsp; · &nbsp; x̂ = decoder(z) ≈ clean x":"z = encoder(x) &nbsp; · &nbsp; x̂ = decoder(z) ≈ x";if(math)math.innerHTML=`The ${conceptState.features}-value code retains <b>${Math.round(result.energy*100)}%</b> of the prepared feature energy. Error against the ${conceptState.task==="denoise"?"clean":"observed"} target is <b>${fmt(error,4)}</b>.`;if(readout)readout.innerHTML=conceptState.task==="denoise"?`<strong>Denoising objective:</strong> output-to-clean error is ${fmt(cleanError,4)}. ${gain>=0?`That improves on the noisy input by ${gain.toFixed(0)}%.`:`At this bottleneck it is ${Math.abs(gain).toFixed(0)}% higher than the noisy input; try retaining more features.`}`:`<strong>Reconstruction objective:</strong> the decoder is asked to reproduce the model input, including any noise it has enough capacity to preserve.`;
    }else{
      const sorted=[...latentData.decoded.weights].sort((a,b)=>b.value-a.value),nearest=AE_PLAY_ANCHORS[sorted[0].key];if(formula)formula.innerHTML="x̂ = decoder(z₁, z₂) &nbsp; · &nbsp; nearby z → nearby outputs";if(math)math.innerHTML=`The selected code is z = (<b>${fmt(conceptState.latentX,2)}</b>, <b>${fmt(conceptState.latentY,2)}</b>). Its strongest prepared contribution is <b>${nearest.label}</b> at ${Math.round(sorted[0].value*100)}%.`;if(readout)readout.innerHTML=`<strong>Latent interpolation:</strong> the blue ring is the encoded input; the purple point is the code you control. Drag between clusters and watch the decoded image blend continuously.`;
    }
  };
  conceptRender=()=>{
    const input=observed(),target=conceptState.task==="denoise"?conceptState.clean:input,result=aePlayReconstruct(input,conceptState.features,conceptState.task,conceptState.noise),size=conceptSize(),context=conceptContext;context.clearRect(0,0,size.width,size.height);context.fillStyle="#f8fbfd";context.fillRect(0,0,size.width,size.height);let latentData=null;
    if(conceptState.view==="pipeline"){conceptElements.plotLabel.textContent="Encoder → bottleneck → decoder";conceptElements.plotTitle.textContent="What survives compression?";pipelineRender(context,size,input,result,target);const targetError=aePlayMse(result.reconstruction,target),cleanError=aePlayMse(result.reconstruction,conceptState.clean),noiseError=aePlayMse(input,conceptState.clean);metricCards([{label:"Target MSE",value:fmt(targetError,4),note:conceptState.task==="denoise"?"Output versus clean":"Output versus model input"},{label:"Compression",value:`${conceptState.features} / 64`,note:`${Math.round(conceptState.features/64*100)}% as many values`},{label:"Energy retained",value:`${Math.round(result.energy*100)}%`,note:"In retained features"},{label:"Clean-image MSE",value:fmt(cleanError,4),note:`Noisy input ${fmt(noiseError,4)}`}])}
    else{conceptElements.plotLabel.textContent="Latent-space explorer";conceptElements.plotTitle.textContent="Nearby codes create related outputs";latentData=latentRender(context,size,input);const sorted=[...latentData.decoded.weights].sort((a,b)=>b.value-a.value),encodedDistance=Math.hypot(conceptState.latentX-latentData.encoded.x,conceptState.latentY-latentData.encoded.y);metricCards([{label:"Horizontal code z₁",value:fmt(conceptState.latentX,2),note:"Left ↔ right"},{label:"Vertical code z₂",value:fmt(conceptState.latentY,2),note:"Bottom ↔ top"},{label:"Strongest region",value:AE_PLAY_ANCHORS[sorted[0].key].label,note:`${Math.round(sorted[0].value*100)}% contribution`},{label:"From encoded input",value:fmt(encodedDistance,2),note:"Distance in latent space"}])}
    updateExplanation(input,result,target,latentData);
  };
  const setView=view=>{conceptState.view=view;updateControlState();updateLegend();conceptElements.help.textContent=view==="pipeline"?"Click the model-input pixels to change the clean drawing. Compare the clean reference, noisy observation and decoded output.":"Drag the purple code through the map. The blue ring marks where the current input is encoded; arrow keys move the selected code.";conceptCanvas.style.cursor=view==="latent"?"grab":"pointer";conceptRender()};
  const placeEncoded=()=>{const encoded=aePlayLatentEncode(observed());conceptState.latentX=encoded.x;conceptState.latentY=encoded.y;syncLatentControls();conceptRender()};
  const updateLatentFromPointer=event=>{const area=conceptState.latentArea,position=conceptPosition(event);if(!area)return;conceptState.latentX=clamp((position.x-area.x)/area.w*2-1,-1,1);conceptState.latentY=clamp(1-(position.y-area.y)/area.h*2,-1,1);syncLatentControls();conceptRender()};
  document.querySelectorAll('input[name="aeView"]').forEach(input=>input.addEventListener("change",event=>{if(event.target.checked)setView(event.target.value)}));
  $("#aePattern").value=conceptState.pattern;$("#aePattern").addEventListener("change",event=>{conceptState.pattern=event.target.value;if(conceptState.pattern!=="custom"){conceptState.lastPattern=conceptState.pattern;conceptState.clean=aePlayPattern(conceptState.pattern);const anchor=AE_PLAY_ANCHORS[conceptState.pattern];conceptState.latentX=anchor.x;conceptState.latentY=anchor.y;syncLatentControls()}conceptRender()});
  bindRange("aeNoise",value=>conceptState.noise=value,2);bindRange("aeCode",value=>conceptState.features=value,0);$("#aeTask").value=conceptState.task;$("#aeTask").addEventListener("change",event=>{conceptState.task=event.target.value;conceptRender()});bindRange("aeLatentX",value=>conceptState.latentX=value,2);bindRange("aeLatentY",value=>conceptState.latentY=value,2);
  $("#aeNoiseNew").addEventListener("click",()=>{conceptState.seed++;conceptRender()});$("#aeUseEncoded").addEventListener("click",placeEncoded);$("#aeClear").addEventListener("click",()=>{conceptState.pattern="custom";conceptState.clean=Array.from({length:8},()=>Array(8).fill(0));$("#aePattern").value="custom";conceptRender()});$("#aeReset").addEventListener("click",()=>{conceptState.pattern=conceptState.lastPattern;conceptState.clean=aePlayPattern(conceptState.lastPattern);conceptState.features=6;conceptState.noise=.25;conceptState.task="reconstruct";conceptState.seed=881;const anchor=AE_PLAY_ANCHORS[conceptState.lastPattern];conceptState.latentX=anchor.x;conceptState.latentY=anchor.y;$("#aePattern").value=conceptState.pattern;$("#aeCode").value=6;$("#aeCodeOut").textContent="6";$("#aeNoise").value=.25;$("#aeNoiseOut").textContent="0.25";$("#aeTask").value="reconstruct";syncLatentControls();conceptRender()});
  conceptPointerDown=event=>{const position=conceptPosition(event);if(conceptState.view==="pipeline"){const area=conceptState.inputArea;if(!area||position.x<area.x||position.x>=area.x+area.cell*8||position.y<area.y||position.y>=area.y+area.cell*8)return;const column=Math.floor((position.x-area.x)/area.cell),row=Math.floor((position.y-area.y)/area.cell);conceptState.clean[row][column]=conceptState.clean[row][column]>=.5?0:1;conceptState.pattern="custom";$("#aePattern").value="custom";conceptRender();return}const area=conceptState.latentArea;if(area&&position.x>=area.x&&position.x<=area.x+area.w&&position.y>=area.y&&position.y<=area.y+area.h){conceptState.dragging=true;conceptCanvas.setPointerCapture(event.pointerId);conceptCanvas.style.cursor="grabbing";updateLatentFromPointer(event)}};
  conceptPointerMove=event=>{if(conceptState.dragging)updateLatentFromPointer(event)};conceptPointerUp=event=>{if(conceptState.dragging&&conceptCanvas.hasPointerCapture(event.pointerId))conceptCanvas.releasePointerCapture(event.pointerId);conceptState.dragging=false;conceptCanvas.style.cursor=conceptState.view==="latent"?"grab":"pointer"};conceptKeyDown=event=>{if(conceptState.view!=="latent"||!["ArrowLeft","ArrowRight","ArrowUp","ArrowDown"].includes(event.key))return;event.preventDefault();if(event.key==="ArrowLeft")conceptState.latentX-=.05;if(event.key==="ArrowRight")conceptState.latentX+=.05;if(event.key==="ArrowUp")conceptState.latentY+=.05;if(event.key==="ArrowDown")conceptState.latentY-=.05;conceptState.latentX=clamp(conceptState.latentX,-1,1);conceptState.latentY=clamp(conceptState.latentY,-1,1);syncLatentControls();conceptRender()};
  conceptCanvas.setAttribute("aria-label","Interactive autoencoder reconstruction and latent-space explorer. Draw pixels, or use arrow keys to move the selected latent code.");syncLatentControls();updateControlState();updateLegend();conceptRender();
}

CONCEPT_SETUPS.autoencoder=setupAutoencoderPlayground;
