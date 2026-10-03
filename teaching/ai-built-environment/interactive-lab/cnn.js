"use strict";

CONCEPT_META.cnn={
  category:"Deep Learning",
  title:"CNN Playground",
  summary:"Trace an image through convolution, activation and pooling while editing both pixels and filter weights.",
  question:"How does one small filter turn local pixels into reusable visual features?"
};

const CNN_PLAY_KERNELS={
  vertical:[[-1,0,1],[-1,0,1],[-1,0,1]],
  horizontal:[[-1,-1,-1],[0,0,0],[1,1,1]],
  diagonal:[[-1,-1,0],[-1,0,1],[0,1,1]],
  blur:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
  sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]]
};

function cnnPlayClone(matrix){return matrix.map(row=>[...row])}
function cnnPlayPattern(name){
  const size=7,grid=Array.from({length:size},()=>Array(size).fill(0));
  if(name==="edge")for(let row=0;row<size;row++)for(let column=4;column<size;column++)grid[row][column]=1;
  if(name==="cross")for(let index=0;index<size;index++){grid[3][index]=1;grid[index][3]=1}
  if(name==="corner")for(let index=1;index<6;index++){grid[1][index]=1;grid[index][1]=1}
  if(name==="checker")for(let row=0;row<size;row++)for(let column=0;column<size;column++)grid[row][column]=(row+column)%2;
  if(name==="diagonal")for(let row=0;row<size;row++)for(let column=0;column<size;column++)if(Math.abs(row-column)<=1)grid[row][column]=1;
  return grid;
}
function cnnPlayPad(matrix,padding){
  if(!padding)return cnnPlayClone(matrix);const size=matrix.length+padding*2;
  return Array.from({length:size},(_,row)=>Array.from({length:size},(_,column)=>row<padding||column<padding||row>=size-padding||column>=size-padding?0:matrix[row-padding][column-padding]));
}
function cnnPlayConvolve(pixels,kernel,stride,padding){
  const padded=cnnPlayPad(pixels,padding),rows=Math.floor((padded.length-3)/stride)+1,columns=Math.floor((padded[0].length-3)/stride)+1,map=[];
  for(let row=0;row<rows;row++){
    const line=[];
    for(let column=0;column<columns;column++){
      let sum=0;for(let kr=0;kr<3;kr++)for(let kc=0;kc<3;kc++)sum+=padded[row*stride+kr][column*stride+kc]*kernel[kr][kc];line.push(sum);
    }
    map.push(line);
  }
  return {map,padded};
}
function cnnPlayActivate(map,name){return map.map(row=>row.map(value=>name==="relu"?Math.max(0,value):value))}
function cnnPlayPool(map,type){
  if(type==="none")return null;const output=[];
  for(let row=0;row<map.length;row+=2){const line=[];for(let column=0;column<map[0].length;column+=2){const values=[];for(let r=row;r<Math.min(row+2,map.length);r++)for(let c=column;c<Math.min(column+2,map[0].length);c++)values.push(map[r][c]);line.push(type==="avg"?values.reduce((sum,value)=>sum+value,0)/values.length:Math.max(...values))}output.push(line)}
  return output;
}
function cnnPlayColour(value,maxAbs,input=false,padding=false){
  if(padding)return "#e7edf2";
  if(input){const amount=clamp(value,0,1);return `rgb(${Math.round(246-170*amount)},${Math.round(249-126*amount)},${Math.round(251-80*amount)})`}
  const amount=clamp(Math.abs(value)/(maxAbs||1),0,1),target=value>=0?[36,110,170]:[226,120,53],base=[247,249,250],mix=.12+.78*amount;
  return `rgb(${target.map((channel,index)=>Math.round(base[index]*(1-mix)+channel*mix)).join(",")})`;
}

function setupCnnPlayground(){
  setConcept(
    CONCEPT_META.cnn,
    "CNN playground",
    "Image → convolution → activation → pooling",
    "Click image pixels to draw. Click an output cell to move the filter there. Edit any kernel weight, then scan the same filter across the image.",
    "<strong>Start here</strong><p>Choose <b>Pipeline</b> and move one window at a time. Match the highlighted 3 × 3 pixels to the live multiplication before comparing the full feature maps.</p>"
  );
  const kernelInputs=Array.from({length:9},(_,index)=>`<input type="number" min="-5" max="5" step="0.1" value="${CNN_PLAY_KERNELS.vertical[Math.floor(index/3)][index%3]}" data-cnn-weight="${index}" aria-label="Kernel row ${Math.floor(index/3)+1}, column ${index%3+1}">`).join("");
  conceptElements.controls.innerHTML=
    '<span class="field-label">Visualization</span><div class="train-view-switch cnn-view-switch" role="radiogroup" aria-label="CNN visualization"><label><input id="cnnViewPipeline" type="radio" name="cnnView" value="pipeline" checked><span>Pipeline</span></label><label><input id="cnnViewStack" type="radio" name="cnnView" value="stack"><span>Feature stack</span></label></div>'+
    '<div class="train-control-section"><span class="train-control-heading">1 · Input image</span>'+selectControl("cnnPattern","Pixel pattern",[["edge","Vertical edge"],["cross","Cross"],["corner","Corner"],["checker","Checkerboard"],["diagonal","Diagonal stroke"],["custom","Custom drawing"]])+'</div>'+
    '<div class="train-control-section"><span class="train-control-heading">2 · Filter</span>'+selectControl("cnnKernel","Kernel preset",[["vertical","Vertical edge"],["horizontal","Horizontal edge"],["diagonal","Diagonal edge"],["blur","Blur"],["sharpen","Sharpen"],["custom","Custom weights"]])+'<span class="field-label">Editable 3 × 3 weights</span><div class="cnn-kernel-editor">'+kernelInputs+'</div></div>'+
    '<div class="train-control-section"><span class="train-control-heading">3 · Geometry</span>'+selectControl("cnnPadding","Padding",[["0","Valid · no padding"],["1","Zero padding · one cell"]])+selectControl("cnnStride","Stride",[["1","One pixel"],["2","Two pixels"]])+'</div>'+
    '<div class="train-control-section"><span class="train-control-heading">4 · Layer operations</span>'+selectControl("cnnActivation","Activation",[["relu","ReLU"],["none","No activation"]])+selectControl("cnnPool","Pooling",[["max","2 × 2 max pooling"],["avg","2 × 2 average pooling"],["none","No pooling"]])+'</div>'+
    '<div class="control"><label for="cnnPosition">Current window <em>p</em><output id="cnnPositionOut">1</output></label><input id="cnnPosition" type="range" min="0" max="24" step="1" value="0"></div>';
  conceptElements.actions.innerHTML='<button class="button primary" id="cnnNext" type="button">Next window</button><button class="button" id="cnnAuto" type="button">▶ Scan</button><button class="button quiet" id="cnnReset" type="button">Reset image</button><button class="button quiet" id="cnnClear" type="button">Clear image</button>';
  legend([{label:"Positive response",color:conceptPalette.blue},{label:"Negative response",color:conceptPalette.orange},{label:"Active receptive field",color:conceptPalette.ink},{label:"Padding",color:"#dfe6ec"}]);
  setExplanation(
    '<p>A convolutional filter looks at one small neighbourhood at a time. Its nine weights are reused at every location, producing a feature map that records where the same pattern appears.</p><div class="try-card"><strong>Try this</strong><p>Use the cross image. Compare vertical and horizontal filters, then edit one weight. Ask which responses change and why every location is affected.</p></div><p class="concept-readout" id="cnnReadout"><strong>Receptive field:</strong> the outlined input window produces the outlined feature-map cell.</p>',
    '<div class="formula compact">z[r,c] = ΣᵢΣⱼ image[r+i,c+j] · kernel[i,j]</div><p id="cnnMath"></p>',
    '<ul class="plain-list"><li><b>Weight sharing:</b> the same nine weights inspect every image location.</li><li><b>Padding:</b> artificial border values let the filter reach image edges and change output size.</li><li><b>Stride:</b> larger steps reduce spatial resolution by skipping locations.</li><li><b>ReLU:</b> negative responses become zero while positive responses remain.</li><li><b>Pooling:</b> nearby activations are summarised, increasing the effective receptive field while reducing spatial detail.</li><li>Real CNNs learn many filters from data; this playground isolates the forward operations without expensive training.</li></ul>'
  );

  conceptState={view:"pipeline",pattern:"edge",lastPattern:"edge",pixels:cnnPlayPattern("edge"),kernelName:"vertical",kernel:cnnPlayClone(CNN_PLAY_KERNELS.vertical),padding:0,stride:1,activation:"relu",pool:"max",position:2,inputArea:null,mapAreas:[]};
  const compute=(kernel=conceptState.kernel)=>{const convolution=cnnPlayConvolve(conceptState.pixels,kernel,conceptState.stride,conceptState.padding),activated=cnnPlayActivate(convolution.map,conceptState.activation),pooled=cnnPlayPool(activated,conceptState.pool);return {...convolution,raw:convolution.map,activated,pooled}};
  const syncKernelInputs=()=>document.querySelectorAll("[data-cnn-weight]").forEach(input=>{const index=Number(input.dataset.cnnWeight),value=conceptState.kernel[Math.floor(index/3)][index%3];input.value=String(Math.round(value*1000)/1000);input.style.background=value>0?"#e4f0f8":value<0?"#fff0e6":"#fff"});
  const updatePositionLimit=()=>{const result=compute(),maximum=result.raw.length*result.raw[0].length-1;conceptState.position=clamp(conceptState.position,0,maximum);const control=$("#cnnPosition");control.max=maximum;control.value=conceptState.position;$("#cnnPositionOut").textContent=`${conceptState.position+1} / ${maximum+1}`};
  const matrixGeometry=(matrix,centerX,y,maxWidth,maxHeight,maxCell=28)=>{const rows=matrix.length,columns=matrix[0].length,cell=Math.max(8,Math.min(maxCell,maxWidth/columns,maxHeight/rows)),width=columns*cell,height=rows*cell;return {x:centerX-width/2,y,w:width,h:height,cell,rows,columns}};
  const drawMatrix=(context,matrix,geometry,label,options={})=>{
    const flat=matrix.flat(),maxAbs=Math.max(.0001,...flat.map(Math.abs));context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign="center";context.fillText(label,geometry.x+geometry.w/2,geometry.y-10);
    matrix.forEach((line,row)=>line.forEach((value,column)=>{
      const padded=Boolean(options.padding&&((row<options.padding)||(column<options.padding)||(row>=matrix.length-options.padding)||(column>=matrix[0].length-options.padding)));
      context.fillStyle=cnnPlayColour(value,maxAbs,options.input,padded);context.fillRect(geometry.x+column*geometry.cell,geometry.y+row*geometry.cell,geometry.cell,geometry.cell);context.strokeStyle=padded?"#bdc9d3":"#c3cfd8";context.lineWidth=1;context.strokeRect(geometry.x+column*geometry.cell,geometry.y+row*geometry.cell,geometry.cell,geometry.cell);
      if(geometry.cell>=18&&!padded){context.fillStyle=options.input?(value>.55?"#fff":conceptPalette.ink):(Math.abs(value)/maxAbs>.58?"#fff":conceptPalette.ink);context.font="700 9px system-ui";context.fillText(Math.abs(value)<10?fmt(value,1):fmt(value,0),geometry.x+(column+.5)*geometry.cell,geometry.y+(row+.63)*geometry.cell)}
    }));
    if(options.selected){context.strokeStyle=conceptPalette.ink;context.lineWidth=3;context.strokeRect(geometry.x+options.selected.column*geometry.cell+1.5,geometry.y+options.selected.row*geometry.cell+1.5,geometry.cell-3,geometry.cell-3)}
    if(options.block){context.strokeStyle=conceptPalette.ink;context.lineWidth=3;context.strokeRect(geometry.x+options.block.column*geometry.cell+1.5,geometry.y+options.block.row*geometry.cell+1.5,options.block.columns*geometry.cell-3,options.block.rows*geometry.cell-3)}
    if(options.poolBlock){context.save();context.setLineDash([4,3]);context.strokeStyle=conceptPalette.purple;context.lineWidth=2.5;context.strokeRect(geometry.x+options.poolBlock.column*geometry.cell+1.5,geometry.y+options.poolBlock.row*geometry.cell+1.5,options.poolBlock.columns*geometry.cell-3,options.poolBlock.rows*geometry.cell-3);context.restore()}
    if(options.kind)conceptState.mapAreas.push({geometry,kind:options.kind});return geometry;
  };
  const arrow=(context,from,to,label="")=>{const x1=from.x+from.w+5,y1=from.y+from.h/2,x2=to.x-7,y2=to.y+to.h/2;context.beginPath();context.moveTo(x1,y1);context.lineTo(x2,y2);context.strokeStyle="#8196a7";context.lineWidth=1.7;context.stroke();const angle=Math.atan2(y2-y1,x2-x1);context.beginPath();context.moveTo(x2,y2);context.lineTo(x2-6*Math.cos(angle-.45),y2-6*Math.sin(angle-.45));context.lineTo(x2-6*Math.cos(angle+.45),y2-6*Math.sin(angle+.45));context.closePath();context.fillStyle="#8196a7";context.fill();if(label){context.font="700 9px system-ui";context.textAlign="center";context.fillStyle=conceptPalette.muted;context.fillText(label,(x1+x2)/2,(y1+y2)/2-6)}};
  const drawCalculation=(context,size,windowValues,rawValue,activeValue)=>{
    const w=size.width,wide=w>=560,bandHeight=wide?165:125,y=size.height-bandHeight;context.fillStyle="#edf3f6";context.fillRect(0,y,w,bandHeight);context.fillStyle=conceptPalette.ink;context.font="800 10px system-ui";context.textAlign="left";context.fillText("CURRENT 3 × 3 DOT PRODUCT",14,y+19);
    if(wide){const gap=5,boxW=Math.min(60,(w-165-gap*8)/9),startX=14;windowValues.forEach((item,index)=>{const x=startX+index*(boxW+gap);context.fillStyle="#fff";context.strokeStyle="#ccd8e1";context.lineWidth=1;context.beginPath();context.roundRect(x,y+30,boxW,54,6);context.fill();context.stroke();context.fillStyle=conceptPalette.muted;context.font="700 9px system-ui";context.textAlign="center";context.fillText(`${fmt(item.pixel,0)} × ${fmt(item.weight,1)}`,x+boxW/2,y+49);context.fillStyle=item.product>=0?conceptPalette.blue:conceptPalette.orange;context.font="800 11px system-ui";context.fillText(`= ${fmt(item.product,1)}`,x+boxW/2,y+69)});context.fillStyle=conceptPalette.ink;context.font="800 12px system-ui";context.textAlign="right";context.fillText(`Σ = ${fmt(rawValue,2)}`,w-14,y+49);context.fillText(conceptState.activation==="relu"?`ReLU = ${fmt(activeValue,2)}`:`output = ${fmt(activeValue,2)}`,w-14,y+72)}
    else{context.fillStyle=conceptPalette.muted;context.font="700 10px system-ui";context.textAlign="left";const products=windowValues.map(item=>fmt(item.product,1)).join(" + ");context.fillText(products,14,y+43,w-28);context.fillStyle=conceptPalette.ink;context.font="800 12px system-ui";context.fillText(`Sum = ${fmt(rawValue,2)}  →  ${conceptState.activation==="relu"?"ReLU":"output"} = ${fmt(activeValue,2)}`,14,y+67,w-28);context.fillStyle=conceptPalette.muted;context.font="700 10px system-ui";context.fillText("Open Live math below for every pixel × weight term.",14,y+93,w-28)}
  };
  const drawPipeline=(context,size,result,row,column,windowValues,rawValue,activeValue)=>{
    const mobile=size.width<560,poolRow=Math.floor(row/2),poolColumn=Math.floor(column/2);conceptState.mapAreas=[];
    if(!mobile){
      const centers=[.12,.31,.49,.69,.89].map(value=>size.width*value),top=66,maxWidth=size.width*.155,maxHeight=205;
      const inputGeom=matrixGeometry(result.padded,centers[0],top,maxWidth,maxHeight,24),kernelGeom=matrixGeometry(conceptState.kernel,centers[1],top+34,maxWidth,140,30),rawGeom=matrixGeometry(result.raw,centers[2],top,maxWidth,maxHeight,24),activeGeom=matrixGeometry(result.activated,centers[3],top,maxWidth,maxHeight,24),poolGeom=result.pooled?matrixGeometry(result.pooled,centers[4],top+24,maxWidth,maxHeight-25,28):{x:centers[4]-45,y:top+72,w:90,h:48};
      arrow(context,inputGeom,kernelGeom,"dot");arrow(context,kernelGeom,rawGeom,"scan");arrow(context,rawGeom,activeGeom,conceptState.activation==="relu"?"ReLU":"identity");arrow(context,activeGeom,poolGeom,conceptState.pool==="none"?"skip":conceptState.pool);
      drawMatrix(context,result.padded,inputGeom,conceptState.padding?`PADDED INPUT · ${result.padded.length}×${result.padded.length}`:"INPUT · 7×7",{input:true,padding:conceptState.padding,block:{row:row*conceptState.stride,column:column*conceptState.stride,rows:3,columns:3},kind:"input"});
      drawMatrix(context,conceptState.kernel,kernelGeom,"SHARED FILTER · 3×3");drawMatrix(context,result.raw,rawGeom,`CONVOLUTION · ${result.raw.length}×${result.raw[0].length}`,{selected:{row,column},kind:"raw"});
      drawMatrix(context,result.activated,activeGeom,conceptState.activation==="relu"?"RELU ACTIVATION":"NO ACTIVATION",{selected:{row,column},poolBlock:result.pooled?{row:poolRow*2,column:poolColumn*2,rows:Math.min(2,result.activated.length-poolRow*2),columns:Math.min(2,result.activated[0].length-poolColumn*2)}:null,kind:"activated"});
      if(result.pooled)drawMatrix(context,result.pooled,poolGeom,conceptState.pool==="max"?"MAX POOL":"AVERAGE POOL",{selected:{row:poolRow,column:poolColumn},kind:"pool"});else{context.fillStyle="#f4f7f9";context.strokeStyle="#c8d4dd";context.beginPath();context.roundRect(poolGeom.x,poolGeom.y,poolGeom.w,poolGeom.h,8);context.fill();context.stroke();context.fillStyle=conceptPalette.muted;context.font="800 10px system-ui";context.textAlign="center";context.fillText("POOLING OFF",poolGeom.x+poolGeom.w/2,poolGeom.y+28)}
    }else{
      const inputGeom=matrixGeometry(result.padded,size.width*.2,59,size.width*.37,130,19),kernelGeom=matrixGeometry(conceptState.kernel,size.width*.56,78,size.width*.25,95,30),rawGeom=matrixGeometry(result.raw,size.width*.84,65,size.width*.29,120,22),activeGeom=matrixGeometry(result.activated,size.width*.28,247,size.width*.42,105,21),poolGeom=result.pooled?matrixGeometry(result.pooled,size.width*.73,260,size.width*.36,92,27):{x:size.width*.61,y:280,w:size.width*.27,h:48};
      arrow(context,inputGeom,kernelGeom);arrow(context,kernelGeom,rawGeom);arrow(context,activeGeom,poolGeom);
      drawMatrix(context,result.padded,inputGeom,conceptState.padding?"PADDED INPUT":"INPUT",{input:true,padding:conceptState.padding,block:{row:row*conceptState.stride,column:column*conceptState.stride,rows:3,columns:3},kind:"input"});drawMatrix(context,conceptState.kernel,kernelGeom,"FILTER");drawMatrix(context,result.raw,rawGeom,"CONVOLUTION",{selected:{row,column},kind:"raw"});
      drawMatrix(context,result.activated,activeGeom,conceptState.activation==="relu"?"RELU":"ACTIVATION",{selected:{row,column},poolBlock:result.pooled?{row:poolRow*2,column:poolColumn*2,rows:Math.min(2,result.activated.length-poolRow*2),columns:Math.min(2,result.activated[0].length-poolColumn*2)}:null,kind:"activated"});
      if(result.pooled)drawMatrix(context,result.pooled,poolGeom,conceptState.pool==="max"?"MAX POOL":"AVG POOL",{selected:{row:poolRow,column:poolColumn},kind:"pool"});else{context.fillStyle=conceptPalette.muted;context.font="800 10px system-ui";context.textAlign="center";context.fillText("POOLING OFF",size.width*.73,305)}
    }
    drawCalculation(context,size,windowValues,rawValue,activeValue);
  };
  const drawStack=(context,size,result,row,column)=>{
    conceptState.mapAreas=[];const mobile=size.width<560,alternate=conceptState.kernelName==="horizontal"?"vertical":"horizontal",filters=[{name:"SELECTED",kernel:conceptState.kernel},{name:alternate.toUpperCase(),kernel:CNN_PLAY_KERNELS[alternate]},{name:"DIAGONAL",kernel:CNN_PLAY_KERNELS.diagonal},{name:"BLUR",kernel:CNN_PLAY_KERNELS.blur}],maps=filters.map(filter=>cnnPlayActivate(cnnPlayConvolve(conceptState.pixels,filter.kernel,conceptState.stride,conceptState.padding).map,conceptState.activation));
    context.fillStyle=conceptPalette.muted;context.font="800 10px system-ui";context.textAlign="left";context.fillText("ONE IMAGE · FOUR FILTERS · FOUR FEATURE MAPS",14,22);
    if(!mobile){
      const inputGeom=matrixGeometry(result.padded,size.width*.21,105,size.width*.34,250,28);drawMatrix(context,result.padded,inputGeom,conceptState.padding?"PADDED INPUT":"INPUT IMAGE",{input:true,padding:conceptState.padding,block:{row:row*conceptState.stride,column:column*conceptState.stride,rows:3,columns:3},kind:"input"});
      const centersX=[size.width*.55,size.width*.82],tops=[67,265];maps.forEach((map,index)=>{const geom=matrixGeometry(map,centersX[index%2],tops[Math.floor(index/2)],size.width*.22,135,24);drawMatrix(context,map,geom,`${filters[index].name} MAP`,{selected:{row,column},kind:"stack"})});
      context.beginPath();context.moveTo(inputGeom.x+inputGeom.w+18,inputGeom.y+inputGeom.h/2);context.lineTo(size.width*.45,inputGeom.y+inputGeom.h/2);context.strokeStyle="#8196a7";context.lineWidth=2;context.stroke();
    }else{
      const inputGeom=matrixGeometry(result.padded,size.width/2,57,size.width*.5,155,22);drawMatrix(context,result.padded,inputGeom,conceptState.padding?"PADDED INPUT":"INPUT IMAGE",{input:true,padding:conceptState.padding,block:{row:row*conceptState.stride,column:column*conceptState.stride,rows:3,columns:3},kind:"input"});
      const centersX=[size.width*.27,size.width*.73],tops=[270,375];maps.forEach((map,index)=>{const geom=matrixGeometry(map,centersX[index%2],tops[Math.floor(index/2)],size.width*.38,75,17);drawMatrix(context,map,geom,`${filters[index].name} MAP`,{selected:{row,column},kind:"stack"})});
    }
  };
  const renderMetrics=(result,row,column,rawValue,activeValue)=>{
    const pooled=result.pooled?result.pooled[Math.floor(row/2)][Math.floor(column/2)]:null,receptive=result.pooled?3+conceptState.stride:3;
    metricCards([{label:"Raw convolution",value:fmt(rawValue,3),note:`Output row ${row+1}, column ${column+1}`},{label:"After activation",value:fmt(activeValue,3),note:conceptState.activation==="relu"?"Negative values clipped":"Identity activation"},{label:"Feature-map size",value:`${result.raw.length} × ${result.raw[0].length}`,note:`Padding ${conceptState.padding} · stride ${conceptState.stride}`},{label:result.pooled?"Selected pooled value":"Receptive field",value:result.pooled?fmt(pooled,3):"3 × 3",note:result.pooled?`${receptive} × ${receptive} effective input area`:"Nine shared weights"}]);
    const startRow=row*conceptState.stride-conceptState.padding+1,endRow=startRow+2,startColumn=column*conceptState.stride-conceptState.padding+1,endColumn=startColumn+2,outside=startRow<1||startColumn<1||endRow>7||endColumn>7,readout=$("#cnnReadout");if(readout)readout.innerHTML=`<strong>Receptive field:</strong> output (${row+1}, ${column+1}) sees original-image rows ${startRow}–${endRow} and columns ${startColumn}–${endColumn}. ${outside?"Coordinates outside 1–7 are zero-padding cells. ":""}${result.pooled?`The selected pooled cell combines nearby responses and sees up to a ${receptive} × ${receptive} input area.`:""}`;
  };
  conceptRender=()=>{
    const result=compute(),columns=result.raw[0].length,maximum=result.raw.length*columns-1;conceptElements.plotLabel.textContent=conceptState.view==="pipeline"?"CNN playground":"Filter bank";conceptElements.plotTitle.textContent=conceptState.view==="pipeline"?"Image → convolution → activation → pooling":"One image, several feature detectors";conceptState.position=clamp(conceptState.position,0,maximum);const row=Math.floor(conceptState.position/columns),column=conceptState.position%columns,inputRow=row*conceptState.stride,inputColumn=column*conceptState.stride,windowValues=[];
    for(let kr=0;kr<3;kr++)for(let kc=0;kc<3;kc++){const pixel=result.padded[inputRow+kr][inputColumn+kc],weight=conceptState.kernel[kr][kc];windowValues.push({pixel,weight,product:pixel*weight})}
    const rawValue=result.raw[row][column],activeValue=result.activated[row][column],size=conceptSize(),context=conceptContext;context.clearRect(0,0,size.width,size.height);context.fillStyle="#f8fbfd";context.fillRect(0,0,size.width,size.height);
    conceptState.view==="pipeline"?drawPipeline(context,size,result,row,column,windowValues,rawValue,activeValue):drawStack(context,size,result,row,column);renderMetrics(result,row,column,rawValue,activeValue);
    const terms=windowValues.map(item=>`${fmt(item.pixel,0)}×${fmt(item.weight,2)}`),math=$("#cnnMath");if(math)math.innerHTML=`At output (${row+1}, ${column+1}): ${terms.join(" + ")} = <b>${fmt(rawValue,2)}</b>. ${conceptState.activation==="relu"?`ReLU gives <b>${fmt(activeValue,2)}</b>.`:"No activation is applied."}`;
  };
  const advance=()=>{const maximum=Number($("#cnnPosition").max);conceptState.position=(conceptState.position+1)%(maximum+1);$("#cnnPosition").value=conceptState.position;$("#cnnPositionOut").textContent=`${conceptState.position+1} / ${maximum+1}`;conceptRender()};
  document.querySelectorAll('input[name="cnnView"]').forEach(input=>input.addEventListener("change",event=>{if(event.target.checked){conceptState.view=event.target.value;conceptRender()}}));
  $("#cnnPattern").value=conceptState.pattern;$("#cnnPattern").addEventListener("change",event=>{stopConceptAnimation();conceptState.pattern=event.target.value;if(conceptState.pattern!=="custom"){conceptState.lastPattern=conceptState.pattern;conceptState.pixels=cnnPlayPattern(conceptState.pattern)}conceptState.position=0;updatePositionLimit();conceptRender()});
  $("#cnnKernel").value=conceptState.kernelName;$("#cnnKernel").addEventListener("change",event=>{stopConceptAnimation();conceptState.kernelName=event.target.value;if(CNN_PLAY_KERNELS[conceptState.kernelName])conceptState.kernel=cnnPlayClone(CNN_PLAY_KERNELS[conceptState.kernelName]);syncKernelInputs();conceptState.position=0;updatePositionLimit();conceptRender()});
  document.querySelectorAll("[data-cnn-weight]").forEach(input=>input.addEventListener("input",event=>{const index=Number(event.target.dataset.cnnWeight),value=clamp(Number(event.target.value)||0,-5,5);conceptState.kernel[Math.floor(index/3)][index%3]=value;conceptState.kernelName="custom";$("#cnnKernel").value="custom";syncKernelInputs();conceptRender()}));
  $("#cnnPadding").value=String(conceptState.padding);$("#cnnPadding").addEventListener("change",event=>{stopConceptAnimation();conceptState.padding=Number(event.target.value);conceptState.position=0;updatePositionLimit();conceptRender()});
  $("#cnnStride").value=String(conceptState.stride);$("#cnnStride").addEventListener("change",event=>{stopConceptAnimation();conceptState.stride=Number(event.target.value);conceptState.position=0;updatePositionLimit();conceptRender()});
  $("#cnnActivation").value=conceptState.activation;$("#cnnActivation").addEventListener("change",event=>{conceptState.activation=event.target.value;conceptRender()});$("#cnnPool").value=conceptState.pool;$("#cnnPool").addEventListener("change",event=>{conceptState.pool=event.target.value;conceptRender()});
  $("#cnnPosition").addEventListener("input",event=>{conceptState.position=Number(event.target.value);$("#cnnPositionOut").textContent=`${conceptState.position+1} / ${Number(event.target.max)+1}`;conceptRender()});$("#cnnNext").addEventListener("click",()=>{stopConceptAnimation();advance()});
  $("#cnnAuto").addEventListener("click",()=>{if(conceptTimer){stopConceptAnimation();return}$("#cnnAuto").textContent="Pause";conceptTimer=setInterval(advance,window.labDelay?window.labDelay(420):420)});
  $("#cnnReset").addEventListener("click",()=>{stopConceptAnimation();conceptState.pattern=conceptState.lastPattern;conceptState.pixels=cnnPlayPattern(conceptState.lastPattern);$("#cnnPattern").value=conceptState.lastPattern;conceptState.position=0;updatePositionLimit();conceptRender()});
  $("#cnnClear").addEventListener("click",()=>{stopConceptAnimation();conceptState.pattern="custom";conceptState.pixels=Array.from({length:7},()=>Array(7).fill(0));$("#cnnPattern").value="custom";conceptState.position=0;updatePositionLimit();conceptRender()});
  conceptPointerDown=event=>{const position=conceptPosition(event),area=conceptState.inputArea||conceptState.mapAreas.find(item=>item.kind==="input")?.geometry,inputHit=conceptState.mapAreas.find(item=>item.kind==="input"&&position.x>=item.geometry.x&&position.x<item.geometry.x+item.geometry.w&&position.y>=item.geometry.y&&position.y<item.geometry.y+item.geometry.h);if(inputHit){const displayRow=Math.floor((position.y-inputHit.geometry.y)/inputHit.geometry.cell),displayColumn=Math.floor((position.x-inputHit.geometry.x)/inputHit.geometry.cell),row=displayRow-conceptState.padding,column=displayColumn-conceptState.padding;if(row>=0&&row<7&&column>=0&&column<7){conceptState.pixels[row][column]=conceptState.pixels[row][column]>=.5?0:1;conceptState.pattern="custom";$("#cnnPattern").value="custom";conceptRender()}return}const mapHit=conceptState.mapAreas.find(item=>item.kind!=="input"&&position.x>=item.geometry.x&&position.x<item.geometry.x+item.geometry.w&&position.y>=item.geometry.y&&position.y<item.geometry.y+item.geometry.h);if(!mapHit)return;const result=compute(),mapRow=Math.floor((position.y-mapHit.geometry.y)/mapHit.geometry.cell),mapColumn=Math.floor((position.x-mapHit.geometry.x)/mapHit.geometry.cell);if(mapHit.kind==="pool")conceptState.position=Math.min((mapRow*2)*result.raw[0].length+mapColumn*2,result.raw.length*result.raw[0].length-1);else conceptState.position=Math.min(mapRow*result.raw[0].length+mapColumn,result.raw.length*result.raw[0].length-1);updatePositionLimit();conceptRender()};
  conceptKeyDown=event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();stopConceptAnimation();const maximum=Number($("#cnnPosition").max);conceptState.position=(conceptState.position+(event.key==="ArrowRight"?1:-1)+(maximum+1))%(maximum+1);$("#cnnPosition").value=conceptState.position;$("#cnnPositionOut").textContent=`${conceptState.position+1} / ${maximum+1}`;conceptRender()};
  conceptCanvas.style.cursor="pointer";conceptCanvas.setAttribute("aria-label","Editable CNN pipeline. Click input pixels to draw, click feature-map cells to move the receptive field, and use left or right arrow keys to scan.");syncKernelInputs();updatePositionLimit();conceptRender();
}

CONCEPT_SETUPS.cnn=setupCnnPlayground;
