"use strict";

/* Object Detection: IoU, NMS & Anchors. The scenes are self-generated drawings of concrete defects (no photographs);
   every number comes from detection-math.js. Image coordinates are pixels of a 320 × 200 image. */
CONCEPT_META.detection={
  category:"Deep Learning",
  title:"Object Detection: IoU, NMS & Anchors",
  summary:"Score boxes with IoU, remove duplicates with non-maximum suppression, match anchors to objects, decode box offsets and summarise a detector with precision–recall and AP.",
  question:"How does a detector decide which boxes count as finding an object?"
};

const DET_IMAGE={w:320,h:200};
const DET_SCENES={
  iou:[{type:"spall",box:[100,60,200,140]}],
  nms:[{type:"spall",box:[58,48,148,132]},{type:"spall",box:[112,56,212,136]}],
  anchors:[{type:"crack",box:[40,30,70,170],name:"Crack"},{type:"spall",box:[150,60,230,130],name:"Spall"},{type:"stain",box:[250,120,300,180],name:"Rust stain"}],
  pr:[{type:"crack",box:[30,20,62,180]},{type:"spall",box:[110,40,180,100]},{type:"spall",box:[200,30,260,90]},{type:"stain",box:[120,130,190,185]},{type:"stain",box:[250,120,300,175]}]
};
const DET_IOU_DEFAULT={gt:[100,60,200,140],pred:[120,70,215,150]};
/* NMS candidates: two neighbouring spalls, three detections each, plus one weak false alarm. */
const DET_NMS_CANDIDATES=[{box:[60,50,150,130],score:.95},{box:[66,44,158,126],score:.81},{box:[52,58,142,138],score:.62},{box:[110,55,210,135],score:.9},{box:[104,62,204,146],score:.76},{box:[114,52,216,132],score:.55},{box:[240,20,290,60],score:.3}];
const DET_PR_DETECTIONS=[[.96,[112,42,178,102]],[.91,[198,28,262,94]],[.87,[34,30,66,170]],[.82,[108,36,186,104]],[.74,[128,138,200,195]],[.66,[150,90,220,140]],[.58,[244,112,296,168]],[.44,[10,60,40,120]],[.35,[262,126,306,180]],[.22,[70,150,110,190]]].map(([score,box])=>({score,box}));
const DET_SCALE_SETS={small:[24,48,96],medium:[40,80,120],large:[64,112,160]};
const DET_RATIO_SETS={standard:[.5,1,2],wide:[1/3,1,3]};
const DET_GRID={columns:8,rows:5,stride:40};
const DET_COLOURS={gt:conceptPalette.blue,pred:conceptPalette.orange,overlap:"rgba(105,86,165,.42)",muted:"#7d8b97"};

const detSceneCache={};
/* Draws a scene once into an offscreen canvas at twice the image resolution. */
function detScene(key){
  if(detSceneCache[key])return detSceneCache[key];
  const canvas=document.createElement("canvas"),scale=2,random=seededRandom(key.length*7919+131);canvas.width=DET_IMAGE.w*scale;canvas.height=DET_IMAGE.h*scale;
  const c=canvas.getContext("2d");c.scale(scale,scale);c.fillStyle="#c5c9cc";c.fillRect(0,0,DET_IMAGE.w,DET_IMAGE.h);
  for(let i=0;i<1500;i++){const g=150+Math.floor(random()*80);c.fillStyle=`rgba(${g},${g},${g+4},.55)`;c.fillRect(random()*DET_IMAGE.w,random()*DET_IMAGE.h,1+random()*1.4,1+random()*1.4)}
  c.strokeStyle="rgba(255,255,255,.4)";c.lineWidth=1;[66,133].forEach(y=>{c.beginPath();c.moveTo(0,y);c.lineTo(DET_IMAGE.w,y);c.stroke()});
  DET_SCENES[key].forEach((defect,index)=>{
    const [x1,y1,x2,y2]=defect.box,w=x2-x1,h=y2-y1,cx=(x1+x2)/2,cy=(y1+y2)/2,phase=index*1.7+random();
    if(defect.type==="crack"){
      c.strokeStyle="#262a2e";c.lineWidth=2.2;c.lineJoin="round";c.beginPath();const steps=16;let mid=null;
      for(let i=0;i<=steps;i++){const t=i/steps,along=y1+2+t*(h-4),across=x1+w*(.5+.32*Math.sin(t*6.5+phase))+(random()-.5)*w*.12;i?c.lineTo(across,along):c.moveTo(across,along);if(i===Math.round(steps*.45))mid=[across,along]}
      c.stroke();c.lineWidth=1.3;c.beginPath();c.moveTo(...mid);c.lineTo(mid[0]+w*.42,mid[1]+h*.12);c.stroke();
    }else if(defect.type==="spall"){
      const blob=(rx,ry,fill)=>{c.beginPath();for(let i=0;i<=16;i++){const a=i/16*Math.PI*2,r=.82+random()*.18;i?c.lineTo(cx+Math.cos(a)*rx*r,cy+Math.sin(a)*ry*r):c.moveTo(cx+Math.cos(a)*rx*r,cy+Math.sin(a)*ry*r)}c.closePath();c.fillStyle=fill;c.fill()};
      blob(w*.5,h*.5,"#8e8983");blob(w*.34,h*.32,"#726d67");c.strokeStyle="#8a4b2c";c.lineWidth=3;[[-.12,.6],[.16,.5]].forEach(([dy,length])=>{c.beginPath();c.moveTo(cx-w*length/2,cy+h*dy);c.lineTo(cx+w*length/2,cy+h*dy);c.stroke()});
    }else{
      const gradient=c.createRadialGradient(cx,cy-h*.1,2,cx,cy-h*.1,Math.max(w,h)*.55);gradient.addColorStop(0,"rgba(150,85,40,.75)");gradient.addColorStop(1,"rgba(150,85,40,0)");c.fillStyle=gradient;c.beginPath();c.ellipse(cx,cy,w/2,h/2,0,0,Math.PI*2);c.fill();
      c.strokeStyle="rgba(130,70,30,.5)";c.lineWidth=2;c.beginPath();c.moveTo(cx+w*.1,cy);c.lineTo(cx+w*.12,y2);c.stroke();
    }
  });
  detSceneCache[key]=canvas;return canvas;
}

window.LAB_GUIDE_VARIANT=window.LAB_GUIDE_VARIANT||{};
window.LAB_GUIDE_VARIANT.detection=()=>activeConcept==="detection"&&conceptState?conceptState.view:null;

function setupDetection(){
  setConcept(CONCEPT_META.detection,"Intersection over union","How well does a predicted box match the object?","Drag a box to move it, or drag a corner to resize it. Keyboard: Tab to the plot, press 1 for the ground truth or 2 for the prediction, then use the arrow keys to move (Shift + arrows resizes).","<strong>Start here</strong><p>Every later step of a detector, from training targets to evaluation, rests on one number: the IoU between two boxes. Begin with the IoU view, then follow the views from left to right.</p>");
  conceptElements.controls.innerHTML=
    '<span class="field-label">Visualization</span><div class="train-view-switch cnn-view-switch" role="radiogroup" aria-label="Object detection view"><label><input id="detViewIou" type="radio" name="detView" value="iou" checked><span>IoU</span></label></div>'+
    '<div class="train-control-section" data-det-views="iou"><span class="train-control-heading">Boxes</span>'+selectControl("detActiveBox","Box moved by the keyboard",[["pred","Prediction (orange)"],["gt","Ground truth (blue)"]])+rangeControl("detIouThreshold","TP threshold","t",.05,.95,.05,.5)+'<input id="detGtBox" type="hidden" value="'+DET_IOU_DEFAULT.gt.join(",")+'"><input id="detPredBox" type="hidden" value="'+DET_IOU_DEFAULT.pred.join(",")+'"><p class="control-hint">A prediction counts as a true positive (TP) when its IoU with the ground truth is at least t; otherwise it is a false positive (FP).</p></div>';
  conceptElements.actions.innerHTML='<button class="button primary" id="detResetBoxes" type="button" data-det-views="iou">Reset boxes</button><button class="button" id="detCover" type="button" data-det-views="iou">Cover the ground truth</button><button class="button quiet" id="detShift" type="button" data-det-views="iou">Shift prediction →</button>';

  const IOU_EXPLANATION=[
    '<p>A detector outputs boxes, so we need a way to say how well a predicted box matches the true one. <b>Intersection over union</b> divides the area the two boxes share by the area they cover together. Identical boxes score 1, disjoint boxes 0. A prediction is usually counted as finding the object (a true positive) when IoU ≥ 0.5.</p><div class="try-card"><strong>Try this</strong><p>Drag the orange prediction until IoU is just above 0.5 and note how far off it still looks. Then press <b>Cover the ground truth</b>: the prediction now contains the whole object, yet IoU drops to 0.39. Finally raise the TP threshold to 0.75 and see how much tighter a box must be.</p></div>',
    '<div class="formula compact">IoU = area(A ∩ B) / area(A ∪ B) &nbsp; · &nbsp; area(A ∪ B) = area(A) + area(B) − area(A ∩ B)</div><p id="detIouMath"></p>',
    '<ul class="plain-list"><li>The intersection of two axis-aligned boxes is a box: [max(x₁), max(y₁), min(x₂), min(y₂)], or empty when min(x₂) ≤ max(x₁) or min(y₂) ≤ max(y₁).</li><li>Coordinates here are continuous, so a box from x = 100 to 200 is 100 pixels wide. Some older code adds 1 to each width (“inclusive pixels”), which changes IoU slightly for small boxes.</li><li>IoU is scale-invariant: doubling both boxes leaves it unchanged, so a 5-pixel error matters far more on a small crack than on a large spall.</li><li>IoU is used three times in a detector: to label anchors during training, to remove duplicates (NMS) and to decide TP or FP during evaluation. COCO averages AP over IoU thresholds 0.50, 0.55, …, 0.95 to reward tight boxes.</li><li>Training losses such as GIoU and DIoU extend IoU so that non-overlapping boxes still receive a useful gradient.</li></ul>'
  ];
  const EXPLANATIONS={iou:IOU_EXPLANATION};
  const VIEW_TEXT={iou:["Intersection over union","How well does a predicted box match the object?","Drag a box to move it, or drag a corner to resize it. Keyboard: Tab to the plot, press 1 for the ground truth or 2 for the prediction, then use the arrow keys to move (Shift + arrows resizes)."]};

  conceptState={view:"iou",gt:[...DET_IOU_DEFAULT.gt],pred:[...DET_IOU_DEFAULT.pred],iouThreshold:.5,active:"pred",drag:null,image:null,explanationKey:""};
  const parseBox=text=>{const values=String(text).split(",").map(Number);return values.length===4&&values.every(Number.isFinite)&&values[2]>values[0]&&values[3]>values[1]?values:null};
  const clampBox=box=>{const w=clamp(box[2]-box[0],8,DET_IMAGE.w),h=clamp(box[3]-box[1],8,DET_IMAGE.h),x=clamp(box[0],0,DET_IMAGE.w-w),y=clamp(box[1],0,DET_IMAGE.h-h);return [x,y,x+w,y+h].map(value=>Math.round(value))};
  /* Box geometry lives in hidden inputs so lesson links restore it; firing "input" also refreshes the guide and address. */
  const storeBoxes=()=>{[["detGtBox","gt"],["detPredBox","pred"]].forEach(([id,key])=>{const input=$("#"+id);if(input.value!==conceptState[key].join(",")){input.value=conceptState[key].join(",");input.dispatchEvent(new Event("input",{bubbles:true}))}})};
  const imageView=(x,y,w,h)=>{const k=Math.min(w/DET_IMAGE.w,h/DET_IMAGE.h);return {x:x+(w-DET_IMAGE.w*k)/2,y,k,w:DET_IMAGE.w*k,h:DET_IMAGE.h*k}};
  const screenBox=(view,box)=>[view.x+box[0]*view.k,view.y+box[1]*view.k,(box[2]-box[0])*view.k,(box[3]-box[1])*view.k];
  const drawScene=(context,view,key)=>{context.drawImage(detScene(key),view.x,view.y,view.w,view.h);context.strokeStyle="#9fb0bd";context.lineWidth=1;context.strokeRect(view.x,view.y,view.w,view.h)};
  const drawBox=(context,view,box,colour,options={})=>{
    const [x,y,w,h]=screenBox(view,box);context.save();if(options.fill){context.fillStyle=options.fill;context.fillRect(x,y,w,h)}context.setLineDash(options.dash||[]);context.strokeStyle=colour;context.lineWidth=options.width||2;context.globalAlpha=options.alpha??1;context.strokeRect(x,y,w,h);context.restore();
    if(options.label){context.save();context.font="800 10px system-ui";const tw=context.measureText(options.label).width+8,ly=options.labelBelow?y+h:y-15<view.y?y:y-15;context.globalAlpha=options.alpha??1;context.fillStyle=colour;context.fillRect(x,ly,tw,15);context.fillStyle="#fff";context.textAlign="left";context.fillText(options.label,x+4,ly+11);context.restore()}
    if(options.handles){context.fillStyle="#fff";context.strokeStyle=colour;context.lineWidth=1.5;[[x,y],[x+w,y],[x,y+h],[x+w,y+h]].forEach(([hx,hy])=>{context.fillRect(hx-4,hy-4,8,8);context.strokeRect(hx-4,hy-4,8,8)})}
  };
  const panelText=(context,lines,x,y,maxWidth)=>lines.forEach(([text,style="body"],index)=>{context.textAlign="left";context.fillStyle=style==="muted"?conceptPalette.muted:style==="head"?conceptPalette.muted:conceptPalette.ink;context.font=style==="head"?"800 10px system-ui":style==="big"?"800 22px system-ui":style==="strong"?"800 12px system-ui":"700 11px system-ui";context.fillText(text,x,y+index*18+(style==="big"?6:0),maxWidth)});

  const drawIou=(context,size)=>{
    const w=size.width,h=size.height,mobile=w<560,view=mobile?imageView(10,34,w-20,h*.5):imageView(14,40,w*.64,h-70),parts=detIouParts(conceptState.gt,conceptState.pred),tp=parts.iou>=conceptState.iouThreshold;conceptState.image=view;
    context.fillStyle=conceptPalette.muted;context.font="800 10px system-ui";context.textAlign="left";context.fillText("BLUE = GROUND TRUTH (LABELLED)   ·   ORANGE = PREDICTION",14,22,w-28);
    drawScene(context,view,"iou");
    if(parts.box){const [x,y,bw,bh]=screenBox(view,parts.box);context.fillStyle=DET_COLOURS.overlap;context.fillRect(x,y,bw,bh)}
    drawBox(context,view,conceptState.gt,DET_COLOURS.gt,{width:conceptState.active==="gt"?3:2.2,label:"ground truth",handles:conceptState.active==="gt"});
    drawBox(context,view,conceptState.pred,DET_COLOURS.pred,{width:conceptState.active==="pred"?3:2.2,dash:[6,3],label:"prediction",labelBelow:true,handles:conceptState.active==="pred"});
    /* Mini diagram: union in light grey, intersection in purple, then the ratio. */
    const panelX=mobile?14:w*.69,panelY=mobile?view.y+view.h+22:52,panelW=mobile?w-28:w-panelX-12,union=[Math.min(conceptState.gt[0],conceptState.pred[0]),Math.min(conceptState.gt[1],conceptState.pred[1]),Math.max(conceptState.gt[2],conceptState.pred[2]),Math.max(conceptState.gt[3],conceptState.pred[3])];
    const thumbW=mobile?110:Math.min(panelW,160),k=thumbW/(union[2]-union[0]),thumbH=(union[3]-union[1])*k,tx=panelX,ty=panelY+8,thumb=box=>[tx+(box[0]-union[0])*k,ty+(box[1]-union[1])*k,(box[2]-box[0])*k,(box[3]-box[1])*k];
    context.fillStyle=conceptPalette.muted;context.font="800 10px system-ui";context.fillText("UNION (GREY) · INTERSECTION (PURPLE)",panelX,panelY,panelW);
    [conceptState.gt,conceptState.pred].forEach(box=>{context.fillStyle="#dfe6ec";context.fillRect(...thumb(box))});if(parts.box){context.fillStyle="rgba(105,86,165,.75)";context.fillRect(...thumb(parts.box))}
    context.strokeStyle=DET_COLOURS.gt;context.lineWidth=1.5;context.strokeRect(...thumb(conceptState.gt));context.save();context.setLineDash([4,2]);context.strokeStyle=DET_COLOURS.pred;context.strokeRect(...thumb(conceptState.pred));context.restore();
    const textX=mobile?tx+thumbW+16:panelX,textY=mobile?panelY+14:ty+Math.min(thumbH,110)+26;
    panelText(context,[[`∩ = ${Math.round(parts.intersection).toLocaleString("en-US")} px²`,"strong"],[`∪ = ${Math.round(parts.union).toLocaleString("en-US")} px²`,"strong"],[`IoU = ${fmt(parts.iou,3)}`,"big"]],textX,textY,mobile?w-textX-12:panelW);
    const pillY=textY+(mobile?70:76);context.fillStyle=tp?conceptPalette.blue:conceptPalette.orange;context.beginPath();context.roundRect(textX,pillY,mobile?172:Math.min(panelW,170),24,12);context.fill();context.fillStyle="#fff";context.font="800 11px system-ui";context.fillText(`${tp?"TRUE POSITIVE":"FALSE POSITIVE"} · t = ${fmt(conceptState.iouThreshold,2)}`,textX+10,pillY+16);
    if(!mobile){const gx=panelX,gy=pillY+46,gw=Math.min(panelW,170);context.fillStyle="#e3eaf0";context.fillRect(gx,gy,gw,10);context.fillStyle=tp?conceptPalette.blue:conceptPalette.orange;context.fillRect(gx,gy,gw*parts.iou,10);context.strokeStyle=conceptPalette.ink;context.lineWidth=2;context.beginPath();context.moveTo(gx+gw*conceptState.iouThreshold,gy-4);context.lineTo(gx+gw*conceptState.iouThreshold,gy+14);context.stroke();context.fillStyle=conceptPalette.muted;context.font="700 9px system-ui";context.fillText("0",gx,gy+24);context.textAlign="right";context.fillText("1",gx+gw,gy+24);context.textAlign="center";context.fillText("t",gx+gw*conceptState.iouThreshold,gy+24)}
    const area=box=>(box[2]-box[0])*(box[3]-box[1]);
    metricCards([{label:"Intersection",value:`${Math.round(parts.intersection).toLocaleString("en-US")} px²`,note:parts.box?`${parts.box[2]-parts.box[0]} × ${parts.box[3]-parts.box[1]} overlap`:"The boxes do not overlap"},{label:"Union",value:`${Math.round(parts.union).toLocaleString("en-US")} px²`,note:`${area(conceptState.gt)} + ${area(conceptState.pred)} − ${Math.round(parts.intersection)}`},{label:"IoU",value:fmt(parts.iou,3),note:"Intersection ÷ union"},{label:"Verdict",value:tp?"TP":"FP",note:`IoU ${tp?"≥":"<"} t = ${fmt(conceptState.iouThreshold,2)}`}]);
    const math=$("#detIouMath"),g=conceptState.gt,p=conceptState.pred;if(math)math.innerHTML=`Ground truth A = [${g.join(", ")}], area ${g[2]-g[0]} × ${g[3]-g[1]} = ${area(g)}. Prediction B = [${p.join(", ")}], area ${p[2]-p[0]} × ${p[3]-p[1]} = ${area(p)}.<br>`+(parts.box?`A ∩ B = [max(${g[0]}, ${p[0]}), max(${g[1]}, ${p[1]}), min(${g[2]}, ${p[2]}), min(${g[3]}, ${p[3]})] = [${parts.box.join(", ")}], area ${parts.box[2]-parts.box[0]} × ${parts.box[3]-parts.box[1]} = <b>${Math.round(parts.intersection)}</b>.`:"The boxes do not overlap, so the intersection is empty (area <b>0</b>).")+`<br>Union = ${area(g)} + ${area(p)} − ${Math.round(parts.intersection)} = <b>${Math.round(parts.union)}</b>, so IoU = ${Math.round(parts.intersection)} / ${Math.round(parts.union)} = <b>${fmt(parts.iou,3)}</b>, which is ${tp?"≥":"<"} t = ${fmt(conceptState.iouThreshold,2)}: a <b>${tp?"true":"false"} positive</b>.`;
  };

  const applyView=()=>{
    const view=conceptState.view;document.querySelectorAll("#conceptControls [data-det-views], #conceptActions [data-det-views]").forEach(element=>{element.hidden=!element.dataset.detViews.split(" ").includes(view)});
    if(conceptState.explanationKey!==view){conceptState.explanationKey=view;setExplanation(...EXPLANATIONS[view])}
    const text=VIEW_TEXT[view];conceptElements.plotLabel.textContent=text[0];conceptElements.plotTitle.textContent=text[1];conceptElements.help.textContent=text[2];
    legend(view==="iou"?[{label:"Ground truth",color:DET_COLOURS.gt},{label:"Prediction",color:DET_COLOURS.pred},{label:"Intersection",color:"#6956a5"}]:[]);
  };
  conceptRender=()=>{const size=conceptSize(),context=conceptContext;context.clearRect(0,0,size.width,size.height);context.fillStyle="#f8fbfd";context.fillRect(0,0,size.width,size.height);if(conceptState.view==="iou")drawIou(context,size)};

  document.querySelectorAll('input[name="detView"]').forEach(input=>input.addEventListener("change",event=>{if(!event.target.checked)return;stopConceptAnimation();conceptState.view=event.target.value;applyView();conceptRender();if(window.labGuideChanged)window.labGuideChanged()}));
  [["detGtBox","gt"],["detPredBox","pred"]].forEach(([id,key])=>$("#"+id).addEventListener("input",event=>{const box=parseBox(event.target.value);if(box){conceptState[key]=clampBox(box);conceptRender()}}));
  $("#detActiveBox").addEventListener("change",event=>{conceptState.active=event.target.value;conceptRender()});
  bindRange("detIouThreshold",value=>conceptState.iouThreshold=value,2);
  $("#detResetBoxes").addEventListener("click",()=>{conceptState.gt=[...DET_IOU_DEFAULT.gt];conceptState.pred=[...DET_IOU_DEFAULT.pred];storeBoxes();conceptRender()});
  $("#detCover").addEventListener("click",()=>{const g=detCenter(conceptState.gt);conceptState.pred=clampBox([g.x-g.w*.8,g.y-g.h*.8,g.x+g.w*.8,g.y+g.h*.8]);storeBoxes();conceptRender()});
  $("#detShift").addEventListener("click",()=>{const p=conceptState.pred;conceptState.pred=clampBox([p[0]+10,p[1],p[2]+10,p[3]]);storeBoxes();conceptRender()});

  /* Pointer: drag a corner handle to resize, or drag inside a box to move it (the prediction wins when both overlap). */
  conceptPointerDown=event=>{
    const view=conceptState.image;if(!view||conceptState.view!=="iou")return;const position=conceptPosition(event),x=(position.x-view.x)/view.k,y=(position.y-view.y)/view.k,tolerance=8/view.k;
    for(const key of ["pred","gt"]){const box=conceptState[key],corners=[[0,1],[2,1],[0,3],[2,3]],corner=corners.find(([cx,cy])=>Math.abs(x-box[cx])<=tolerance&&Math.abs(y-box[cy])<=tolerance);if(corner){conceptState.drag={key,corner,start:{x,y},box:[...box]};break}}
    if(!conceptState.drag)for(const key of ["pred","gt"]){const box=conceptState[key];if(x>=box[0]&&x<=box[2]&&y>=box[1]&&y<=box[3]){conceptState.drag={key,corner:null,start:{x,y},box:[...box]};break}}
    if(conceptState.drag){conceptState.active=conceptState.drag.key;$("#detActiveBox").value=conceptState.active;conceptCanvas.setPointerCapture(event.pointerId);conceptRender()}
  };
  conceptPointerMove=event=>{
    const drag=conceptState.drag,view=conceptState.image;if(!drag||!view)return;const position=conceptPosition(event),dx=(position.x-view.x)/view.k-drag.start.x,dy=(position.y-view.y)/view.k-drag.start.y,box=[...drag.box];
    if(drag.corner){box[drag.corner[0]]+=dx;box[drag.corner[1]]+=dy;const normalised=[Math.min(box[0],box[2]),Math.min(box[1],box[3]),Math.max(box[0],box[2]),Math.max(box[1],box[3])];conceptState[drag.key]=clampBox(normalised)}
    else conceptState[drag.key]=clampBox([box[0]+dx,box[1]+dy,box[2]+dx,box[3]+dy]);
    conceptRender();
  };
  conceptPointerUp=()=>{if(conceptState.drag){conceptState.drag=null;storeBoxes()}};
  conceptKeyDown=event=>{
    if(conceptState.view!=="iou")return;if(event.key==="1"||event.key==="2"){conceptState.active=event.key==="1"?"gt":"pred";$("#detActiveBox").value=conceptState.active;conceptRender();return}
    const moves={ArrowLeft:[-2,0],ArrowRight:[2,0],ArrowUp:[0,-2],ArrowDown:[0,2]}[event.key];if(!moves)return;event.preventDefault();const box=[...conceptState[conceptState.active]];
    if(event.shiftKey){box[2]+=moves[0];box[3]+=moves[1]}else{box[0]+=moves[0];box[2]+=moves[0];box[1]+=moves[1];box[3]+=moves[1]}conceptState[conceptState.active]=clampBox(box);storeBoxes();conceptRender();
  };
  conceptCanvas.style.cursor="move";conceptCanvas.setAttribute("aria-label","Object detection plot. Drag boxes or use the keyboard: 1 or 2 selects a box, arrow keys move it and Shift plus arrow keys resize it.");
  applyView();conceptRender();
}

CONCEPT_SETUPS.detection=setupDetection;
