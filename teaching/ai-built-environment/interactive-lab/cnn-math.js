"use strict";

/* Pure arithmetic for the CNN Playground. No DOM access: the browser loads this file as a classic script
   (functions become globals used by cnn.js) and the Node tests import the same file through the export at
   the end. Conventions are stated beside each function because students see these numbers in Live math. */

const CNN_PLAY_KERNELS={
  vertical:[[-1,0,1],[-1,0,1],[-1,0,1]],
  horizontal:[[-1,-1,-1],[0,0,0],[1,1,1]],
  diagonal:[[-1,-1,0],[-1,0,1],[0,1,1]],
  blur:[[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]],
  sharpen:[[0,-1,0],[-1,5,-1],[0,-1,0]]
};

function cnnPlayClone(matrix){return matrix.map(row=>[...row])}
function cnnZeros(rows,columns){return Array.from({length:rows},()=>Array(columns).fill(0))}

/* Binary teaching patterns. size defaults to the original 7 × 7 grid; the positions scale with size and
   reproduce the original 7 × 7 patterns exactly. */
function cnnPlayPattern(name,size=7){
  const grid=cnnZeros(size,size),middle=Math.floor(size/2);
  if(name==="edge")for(let row=0;row<size;row++)for(let column=middle+1;column<size;column++)grid[row][column]=1;
  if(name==="cross")for(let index=0;index<size;index++){grid[middle][index]=1;grid[index][middle]=1}
  if(name==="corner")for(let index=1;index<size-1;index++){grid[1][index]=1;grid[index][1]=1}
  if(name==="checker")for(let row=0;row<size;row++)for(let column=0;column<size;column++)grid[row][column]=(row+column)%2;
  if(name==="diagonal")for(let row=0;row<size;row++)for(let column=0;column<size;column++)if(Math.abs(row-column)<=1)grid[row][column]=1;
  return grid;
}

/* Self-generated 16 × 16 grayscale scenes (no photograph or external source is used). Values lie in [0, 1];
   as in the binary patterns, 1 is drawn dark ("ink") and 0 light. */
const CNN_SCENES={bridge:"Truss bridge",facade:"Building facade",crack:"Cracked concrete"};
function cnnSeeded(seed){let value=seed>>>0;return()=>{value=(1664525*value+1013904223)>>>0;return value/4294967296}}
function cnnScene(name,size=16){
  const grid=cnnZeros(size,size),random=cnnSeeded({bridge:11,facade:23,crack:37}[name]||5);
  const set=(row,column,value)=>{if(row>=0&&row<size&&column>=0&&column<size)grid[row][column]=value};
  const line=(r0,c0,r1,c1,value)=>{const steps=Math.max(Math.abs(r1-r0),Math.abs(c1-c0));for(let step=0;step<=steps;step++){const t=steps?step/steps:0;set(Math.round(r0+(r1-r0)*t),Math.round(c0+(c1-c0)*t),value)}};
  const fill=(r0,c0,r1,c1,value)=>{for(let row=r0;row<=r1;row++)for(let column=c0;column<=c1;column++)set(row,column,typeof value==="function"?value(row,column):value)};
  if(name==="bridge"){
    fill(0,0,size-1,size-1,row=>.05+.05*row/(size-1));
    fill(13,0,size-1,size-1,(row,column)=>.28+.07*Math.sin(column*1.1+row*2));
    fill(3,2,3,13,.85);line(3,2,8,5,.7);line(8,5,3,8,.7);line(3,8,8,11,.7);line(8,11,3,13,.7);fill(3,2,8,2,.7);fill(3,13,8,13,.7);
    fill(9,0,9,size-1,.92);fill(10,0,10,size-1,.8);
    fill(11,4,size-1,5,.75);fill(11,10,size-1,11,.75);
  }else if(name==="facade"){
    fill(0,0,size-1,size-1,.05);
    fill(2,2,size-1,13,()=>.45+(random()-.5)*.08);fill(2,2,2,13,.8);
    [4,8].forEach(row=>[4,7,10].forEach(column=>fill(row,column,row+1,column+1,.95)));
    fill(12,7,size-1,8,.9);
  }else if(name==="crack"){
    fill(0,0,size-1,size-1,()=>.3+(random()-.5)*.16);
    let column=4;for(let row=0;row<size;row++){set(row,column,.95);if(row===8)line(8,column,11,column+5,.9);column=Math.max(1,Math.min(size-2,column+(random()<.45?1:random()<.7?0:-1)))}
    for(let row=9;row<=13;row++)for(let c=9;c<=13;c++)if(Math.hypot(row-11,c-11)<=2.1&&grid[row][c]<.9)grid[row][c]=.65;
  }
  return grid.map(row=>row.map(value=>Math.round(Math.min(1,Math.max(0,value))*1000)/1000));
}

function cnnPlayPad(matrix,padding){
  if(!padding)return cnnPlayClone(matrix);const size=matrix.length+padding*2;
  return Array.from({length:size},(_,row)=>Array.from({length:size},(_,column)=>row<padding||column<padding||row>=size-padding||column>=size-padding?0:matrix[row-padding][column-padding]));
}

/* Output length for one spatial axis: ⌊(N + 2P − D(F − 1) − 1) / S⌋ + 1 (the PyTorch / Dumoulin–Visin form).
   A result ≤ 0 means the dilated filter does not fit. */
function cnnOutputSize(N,F,P,S,D=1){return Math.floor((N+2*P-D*(F-1)-1)/S)+1}

/* Cross-correlation (the operation deep-learning libraries call convolution) of one channel with one F × F
   kernel. Tap (i, j) of output (r, c) reads padded[r·S + i·D][c·S + j·D]. */
function cnnPlayConvolve(pixels,kernel,stride,padding,dilation=1){
  const padded=cnnPlayPad(pixels,padding),F=kernel.length,rows=Math.max(0,cnnOutputSize(pixels.length,F,padding,stride,dilation)),columns=Math.max(0,cnnOutputSize(pixels[0].length,F,padding,stride,dilation)),map=[];
  for(let row=0;row<rows;row++){
    const line=[];
    for(let column=0;column<columns;column++){
      let sum=0;for(let kr=0;kr<F;kr++)for(let kc=0;kc<F;kc++)sum+=padded[row*stride+kr*dilation][column*stride+kc*dilation]*kernel[kr][kc];line.push(sum);
    }
    map.push(line);
  }
  return {map,padded};
}

/* Multi-channel convolution: one F × F × K filter gives ONE output map, z = Σ_k (channel_k ⋆ slice_k) + b.
   partials holds the per-channel maps before they are summed. */
function cnnMultiChannelConvolve(channels,filter,bias,stride,padding,dilation=1){
  const partials=channels.map((channel,index)=>cnnPlayConvolve(channel,filter[index],stride,padding,dilation).map);
  const rows=partials[0].length,columns=rows?partials[0][0].length:0,map=cnnZeros(rows,columns);
  for(let row=0;row<rows;row++)for(let column=0;column<columns;column++){let sum=0;partials.forEach(partial=>sum+=partial[row][column]);map[row][column]=sum+bias}
  return {map,partials};
}
/* Learnable parameters of a conv layer with L filters of size F × F × K, each with one bias. */
function cnnParamCount(F,K,L){return (F*F*K+1)*L}

/* Activations. Leaky ReLU uses slope 0.1 for negative inputs; ELU uses α = 1. Any other name is identity. */
function cnnActivationValue(name,value){
  if(name==="relu")return Math.max(0,value);
  if(name==="leaky")return value>0?value:.1*value;
  if(name==="elu")return value>0?value:Math.exp(value)-1;
  return value;
}
function cnnPlayActivate(map,name){return map.map(row=>row.map(value=>cnnActivationValue(name,value)))}

/* 2 × 2 pooling with stride 2 in "ceil mode": partial windows at the right and bottom border are kept and
   pooled over the cells they contain, so a 5 × 5 map gives 3 × 3. Average pooling divides by that count. */
function cnnPoolWindows(rows,columns){
  const windows=[];
  for(let row=0;row<rows;row+=2){const line=[];for(let column=0;column<columns;column+=2){const cells=[];for(let r=row;r<Math.min(row+2,rows);r++)for(let c=column;c<Math.min(column+2,columns);c++)cells.push({row:r,column:c});line.push(cells)}windows.push(line)}
  return windows;
}
function cnnPlayPool(map,type){
  if(type==="none")return null;const output=[];
  for(let row=0;row<map.length;row+=2){const line=[];for(let column=0;column<map[0].length;column+=2){const values=[];for(let r=row;r<Math.min(row+2,map.length);r++)for(let c=column;c<Math.min(column+2,map[0].length);c++)values.push(map[r][c]);line.push(type==="avg"?values.reduce((sum,value)=>sum+value,0)/values.length:Math.max(...values))}output.push(line)}
  return output;
}
/* Backward pass of the same pooling. grads has the pooled shape. Max pooling sends each incoming gradient
   only to its window's maximum (ties: the first maximum in row-major order); average pooling splits it
   equally over the cells in the window. Windows do not overlap, so Σ incoming = Σ routed. */
function cnnPoolBackward(map,type,grads){
  const rows=map.length,columns=rows?map[0].length:0,output=cnnZeros(rows,columns),routes=[];
  cnnPoolWindows(rows,columns).forEach((line,pr)=>line.forEach((cells,pc)=>{
    const incoming=grads[pr][pc];let targets;
    if(type==="avg")targets=cells.map(cell=>({...cell,share:incoming/cells.length}));
    else{let best=cells[0];cells.forEach(cell=>{if(map[cell.row][cell.column]>map[best.row][best.column])best=cell});targets=[{...best,share:incoming}]}
    targets.forEach(target=>output[target.row][target.column]+=target.share);routes.push({row:pr,column:pc,incoming,cells,targets});
  }));
  return {grad:output,routes};
}

/* Numerically stable softmax: subtracting the largest logit leaves the probabilities unchanged. */
function cnnSoftmax(logits){const top=Math.max(...logits),exps=logits.map(value=>Math.exp(value-top)),total=exps.reduce((sum,value)=>sum+value,0);return exps.map(value=>value/total)}
/* Classifier head with prototype ("template") weights: w_c = p_c / ‖p_c‖, where p_c is the flattened pooled
   map of class c's prototype image; logit z_c = s · (w_c · x), bias 0. A prototype with no response gets w_c = 0. */
function cnnHeadLogits(features,prototypes,scale=1){
  const weights=prototypes.map(prototype=>{const norm=Math.hypot(...prototype);return norm>1e-12?prototype.map(value=>value/norm):prototype.map(()=>0)});
  const dots=weights.map(weight=>weight.reduce((sum,value,index)=>sum+value*features[index],0));
  return {weights,dots,logits:dots.map(value=>scale*value)};
}

/* Receptive-field arithmetic (Dang Ha The Hien, 2017), extended with dilation through the effective kernel
   size k_eff = d(k − 1) + 1. Layers are {type:"conv"|"pool", k, s, p, d}. Starting from n = input size,
   j = 1, r = 1, start = 0.5:
     n_out = ⌊(n + 2p − k_eff)/s⌋ + 1   (pool: ⌈…⌉, matching the ceil-mode pooling above)
     j_out = j·s,  r_out = r + (k_eff − 1)·j,  start_out = start + ((k_eff − 1)/2 − p)·j */
function cnnReceptiveField(layers,inputSize){
  let current={n:inputSize,j:1,r:1,start:.5};const rows=[{...current,layer:null}];
  layers.forEach(layer=>{
    const d=layer.d||1,p=layer.p||0,kEff=d*(layer.k-1)+1,span=(current.n+2*p-kEff)/layer.s;
    const n=layer.type==="pool"?Math.ceil(span)+1:Math.floor(span)+1;
    current={n,j:current.j*layer.s,r:current.r+(kEff-1)*current.j,start:current.start+((kEff-1)/2-p)*current.j};rows.push({...current,layer});
  });
  return rows;
}
/* Exact input indices that feed one output index, layer by layer (rows and columns are traced separately
   because every operation here is separable). Indices < 0 or ≥ n are padding. levels[0] is the top layer. */
function cnnTraceIndices(layers,sizes,index){
  let indices=[index];const levels=[indices];
  for(let level=layers.length-1;level>=0;level--){
    const layer=layers[level],d=layer.d||1,p=layer.p||0,below=new Set();
    indices.forEach(value=>{for(let tap=0;tap<layer.k;tap++){const source=value*layer.s-p+tap*d;if(layer.type==="pool"&&source>=sizes[level])continue;below.add(source)}});
    indices=[...below].sort((a,b)=>a-b);levels.push(indices);
  }
  return levels;
}

/* Cost of one layer. Parameters include biases; multiply-adds (MACs) count only weight multiplications. */
function cnnConvCost({H,W,Kin,Kout,F,S=1,P=0}){
  const Hout=cnnOutputSize(H,F,P,S),Wout=cnnOutputSize(W,F,P,S);
  return {Hout,Wout,params:(F*F*Kin+1)*Kout,macs:Hout*Wout*F*F*Kin*Kout};
}
/* ResNet-style bottleneck replacing one F × F conv: 1 × 1 reduce Kin→Kmid at the input resolution, F × F
   Kmid→Kmid with the stride and padding, then 1 × 1 expand Kmid→Kout at the output resolution. */
function cnnBottleneckCost({H,W,Kin,Kout,F,S=1,P=0,Kmid}){
  const Hout=cnnOutputSize(H,F,P,S),Wout=cnnOutputSize(W,F,P,S);
  const parts=[{name:"1 × 1 reduce",params:(Kin+1)*Kmid,macs:H*W*Kin*Kmid},{name:`${F} × ${F} conv`,params:(F*F*Kmid+1)*Kmid,macs:Hout*Wout*F*F*Kmid*Kmid},{name:"1 × 1 expand",params:(Kmid+1)*Kout,macs:Hout*Wout*Kmid*Kout}];
  return {Hout,Wout,parts,params:parts.reduce((sum,part)=>sum+part.params,0),macs:parts.reduce((sum,part)=>sum+part.macs,0)};
}
/* MobileNet depthwise-separable conv: one F × F filter per input channel, then a 1 × 1 pointwise conv. */
function cnnSeparableCost({H,W,Kin,Kout,F,S=1,P=0}){
  const Hout=cnnOutputSize(H,F,P,S),Wout=cnnOutputSize(W,F,P,S);
  const parts=[{name:`${F} × ${F} depthwise`,params:(F*F+1)*Kin,macs:Hout*Wout*F*F*Kin},{name:"1 × 1 pointwise",params:(Kin+1)*Kout,macs:Hout*Wout*Kin*Kout}];
  return {Hout,Wout,parts,params:parts.reduce((sum,part)=>sum+part.params,0),macs:parts.reduce((sum,part)=>sum+part.macs,0)};
}
/* Standard ÷ separable multiply-adds = F²·Kout / (F² + Kout); for 3 × 3 with K channels this is 9K/(9 + K). */
function cnnSeparableSpeedup(F,Kout){return F*F*Kout/(F*F+Kout)}

const CNN_MATH={CNN_PLAY_KERNELS,CNN_SCENES,cnnPlayClone,cnnZeros,cnnPlayPattern,cnnScene,cnnPlayPad,cnnOutputSize,cnnPlayConvolve,cnnMultiChannelConvolve,cnnParamCount,cnnActivationValue,cnnPlayActivate,cnnPoolWindows,cnnPlayPool,cnnPoolBackward,cnnSoftmax,cnnHeadLogits,cnnReceptiveField,cnnTraceIndices,cnnConvCost,cnnBottleneckCost,cnnSeparableCost,cnnSeparableSpeedup};
if(typeof module!=="undefined"&&module.exports)module.exports=CNN_MATH;
