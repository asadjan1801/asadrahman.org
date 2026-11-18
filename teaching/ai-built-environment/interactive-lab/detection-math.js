"use strict";

/* Box arithmetic for the Object Detection module. Boxes are corner format [x1, y1, x2, y2] with x2 > x1 and
   y2 > y1, in continuous image coordinates (no "+1 pixel" convention). No DOM access: the browser loads this as a
   classic script and the Node tests import the same file through the export at the end. */

function detArea(box){return Math.max(0,box[2]-box[0])*Math.max(0,box[3]-box[1])}
function detIntersection(a,b){const x1=Math.max(a[0],b[0]),y1=Math.max(a[1],b[1]),x2=Math.min(a[2],b[2]),y2=Math.min(a[3],b[3]);return x2>x1&&y2>y1?[x1,y1,x2,y2]:null}
/* IoU = area(A ∩ B) / area(A ∪ B), with area(A ∪ B) = area(A) + area(B) − area(A ∩ B). */
function detIouParts(a,b){const box=detIntersection(a,b),intersection=box?detArea(box):0,union=detArea(a)+detArea(b)-intersection;return {box,intersection,union,iou:union>0?intersection/union:0}}
function detIou(a,b){return detIouParts(a,b).iou}

/* Greedy non-maximum suppression: visit boxes from the highest score down (ties keep the original order); each box
   that is still available is kept and suppresses every remaining box whose IoU with it is GREATER than the
   threshold. Returns the kept indices and a step-by-step trace. */
function detNms(boxes,scores,threshold){
  const order=scores.map((_,index)=>index).sort((a,b)=>scores[b]-scores[a]||a-b),removed=new Set(),keep=[],steps=[];
  order.forEach(i=>{
    if(removed.has(i))return;keep.push(i);const suppressed=[];
    order.forEach(j=>{if(j===i||removed.has(j)||keep.includes(j))return;const iou=detIou(boxes[i],boxes[j]);if(iou>threshold){removed.add(j);suppressed.push({index:j,iou})}});
    steps.push({kept:i,suppressed});
  });
  return {keep,steps,order};
}

/* Anchor boxes centred on every cell of a columns × rows feature map with the given stride (image pixels per cell).
   For scale s and aspect ratio r = w/h: w = s·√r and h = s/√r, so every anchor of one scale has area s².
   All scales × all ratios are used at each position (Faster R-CNN style; d2l.ai keeps only n + m − 1 of them). */
function detAnchors(columns,rows,stride,scales,ratios){
  const anchors=[];
  for(let row=0;row<rows;row++)for(let column=0;column<columns;column++){
    const cx=(column+.5)*stride,cy=(row+.5)*stride;
    scales.forEach((scale,scaleIndex)=>ratios.forEach((ratio,ratioIndex)=>{const w=scale*Math.sqrt(ratio),h=scale/Math.sqrt(ratio);anchors.push({box:[cx-w/2,cy-h/2,cx+w/2,cy+h/2],column,row,scale,ratio,scaleIndex,ratioIndex})}));
  }
  return anchors;
}
/* Faster R-CNN anchor labels: positive (1) if the anchor's best IoU with any ground truth is ≥ the positive
   threshold, or if it is some ground truth's single best anchor (so every object gets at least one); negative
   (0, background) if its best IoU is below the negative threshold; otherwise ignored (−1) during training. */
function detLabelAnchors(anchorBoxes,gts,positive=.7,negative=.3){
  const ious=anchorBoxes.map(anchor=>gts.map(gt=>detIou(anchor,gt)));
  const result=ious.map(row=>{let gt=-1,best=0;row.forEach((value,index)=>{if(value>best){best=value;gt=index}});return {label:best>=positive?1:best<negative?0:-1,gt,iou:best,forced:false}});
  const bestAnchor=gts.map((_,g)=>{let index=-1,best=0;ious.forEach((row,a)=>{if(row[g]>best){best=row[g];index=a}});return {index,iou:best}});
  bestAnchor.forEach(({index,iou},g)=>{if(index>=0&&result[index].label!==1){result[index]={label:1,gt:g,iou,forced:true}}});
  return {anchors:result,ious,bestAnchor};
}
/* d2l.ai assign_anchor_to_bbox: give every anchor its best ground truth if that IoU ≥ threshold; then repeatedly
   take the largest remaining entry of the anchor × ground-truth IoU matrix, assign that pair and discard its row
   and column, until every ground truth has an anchor. −1 means background. */
function detAssignD2l(anchorBoxes,gts,threshold=.5){
  const ious=anchorBoxes.map(anchor=>gts.map(gt=>detIou(anchor,gt))),map=Array(anchorBoxes.length).fill(-1);
  ious.forEach((row,i)=>{let best=-1,j=-1;row.forEach((value,k)=>{if(value>best){best=value;j=k}});if(best>=threshold)map[i]=j});
  const usedRows=new Set(),usedColumns=new Set();
  for(let step=0;step<gts.length;step++){let best=-1,bi=-1,bj=-1;ious.forEach((row,i)=>{if(usedRows.has(i))return;row.forEach((value,j)=>{if(!usedColumns.has(j)&&value>best){best=value;bi=i;bj=j}})});if(bi<0)break;map[bi]=bj;usedRows.add(bi);usedColumns.add(bj)}
  return map;
}

/* Box-regression targets relative to an anchor (the R-CNN parametrisation), using centres and sizes:
   tx = (x − xa)/wa, ty = (y − ya)/ha, tw = log(w/wa), th = log(h/ha). Optional divisors σ rescale the targets;
   d2l.ai uses σ = (0.1, 0.1, 0.2, 0.2), i.e. multiplies by 10, 10, 5, 5. detDecode inverts detEncode exactly. */
function detCenter(box){return {x:(box[0]+box[2])/2,y:(box[1]+box[3])/2,w:box[2]-box[0],h:box[3]-box[1]}}
function detEncode(anchor,box,sigma=[1,1,1,1]){const a=detCenter(anchor),g=detCenter(box);return [(g.x-a.x)/a.w/sigma[0],(g.y-a.y)/a.h/sigma[1],Math.log(g.w/a.w)/sigma[2],Math.log(g.h/a.h)/sigma[3]]}
function detDecode(anchor,t,sigma=[1,1,1,1]){const a=detCenter(anchor),x=a.x+t[0]*sigma[0]*a.w,y=a.y+t[1]*sigma[1]*a.h,w=a.w*Math.exp(t[2]*sigma[2]),h=a.h*Math.exp(t[3]*sigma[3]);return [x-w/2,y-h/2,x+w/2,y+h/2]}

/* PASCAL VOC matching: detections in descending score order; a detection is a true positive if the ground truth it
   overlaps most has IoU ≥ threshold and has not been matched by a higher-scoring detection; otherwise it is a false
   positive (a duplicate, or a box that is not good enough). */
function detMatch(detections,gts,threshold){
  const order=detections.map((_,index)=>index).sort((a,b)=>detections[b].score-detections[a].score||a-b),matched=new Set();
  return order.map(index=>{let iou=0,gt=-1;gts.forEach((box,j)=>{const value=detIou(detections[index].box,box);if(value>iou){iou=value;gt=j}});const tp=iou>=threshold&&!matched.has(gt);if(tp)matched.add(gt);return {index,score:detections[index].score,iou,gt,tp,duplicate:iou>=threshold&&!tp}});
}
/* Precision and recall after each detection in ranked order. */
function detPrCurve(outcomes,groundTruthCount){let tp=0,fp=0;return outcomes.map(outcome=>{outcome.tp?tp++:fp++;return {score:outcome.score,tp,fp,precision:tp/(tp+fp),recall:groundTruthCount?tp/groundTruthCount:0}})}
/* All-point interpolated AP (PASCAL VOC 2010 onwards): add the points (recall 0) and (recall 1, precision 0), replace
   each precision by the maximum precision at any higher recall (the envelope), and sum the area of the step curve
   wherever recall increases. */
function detAveragePrecision(curve){
  const recall=[0,...curve.map(point=>point.recall),1],precision=[0,...curve.map(point=>point.precision),0];
  for(let i=precision.length-2;i>=0;i--)precision[i]=Math.max(precision[i],precision[i+1]);
  let ap=0;for(let i=1;i<recall.length;i++)if(recall[i]!==recall[i-1])ap+=(recall[i]-recall[i-1])*precision[i];
  return {ap,envelope:recall.map((value,i)=>({recall:value,precision:precision[i]}))};
}
/* PASCAL VOC 2007 11-point AP: mean over t = 0, 0.1, …, 1 of the best precision at recall ≥ t. */
function detElevenPointAp(curve){let sum=0;for(let k=0;k<=10;k++){const values=curve.filter(point=>point.recall>=k/10-1e-12).map(point=>point.precision);sum+=values.length?Math.max(...values):0}return sum/11}

const DET_MATH={detArea,detIntersection,detIouParts,detIou,detNms,detAnchors,detLabelAnchors,detAssignD2l,detCenter,detEncode,detDecode,detMatch,detPrCurve,detAveragePrecision,detElevenPointAp};
if(typeof module!=="undefined"&&module.exports)module.exports=DET_MATH;
