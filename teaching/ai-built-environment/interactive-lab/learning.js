"use strict";

window.labAnimationSpeed=1;
window.labDelay=milliseconds=>Math.max(80,Math.round(milliseconds/(window.labAnimationSpeed||1)));

const learningItems=MODULE_GROUPS.flatMap(group=>group.items);
const misconceptionPrompts={
  linear:"The best-fitting line does not need to pass through every point; it balances all residuals together.",
  gradient:"Gradient descent is not guaranteed to move directly toward the global minimum; it follows local slope information.",
  backprop:"Backpropagation computes gradients. It does not update weights until an optimizer applies those gradients.",
  training:"A lower training loss is not automatically progress if validation loss is rising at the same time.",
  pca:"PCA and t-SNE do not show the same thing: PCA preserves linear variance, while t-SNE prioritizes local neighborhoods.",
  generalization:"A lower training error does not automatically mean a better model on unseen data.",
  logistic:"Changing the decision threshold changes classifications, but it does not retrain the probabilities.",
  knn:"A larger K is not always better; it can wash out a useful local pattern.",
  tree:"A deeper tree is more flexible, but flexibility can fit noise as well as signal.",
  ensembles:"Many learners help when their errors differ; simply duplicating the same learner adds little information.",
  svm:"Support vectors are the influential boundary points, not necessarily unusual data errors.",
  kmeans:"K-means finds compact groups around means; it does not discover every possible cluster shape.",
  evaluation:"Accuracy alone can be misleading when classes or error costs are unequal.",
  arena:"No model wins on every dataset because each method builds in different assumptions.",
  neural:"Nonlinearity, not just extra layers, lets a network build a genuinely nonlinear mapping.",
  cnn:"A kernel is reused across locations; it is not a separate set of weights for every pixel.",
  rnn:"An LSTM gate is a continuous value between 0 and 1, not a simple on/off switch.",
  autoencoder:"A latent coordinate is a learned representation, not automatically a human-readable concept.",
  diffusion:"The reverse process removes predicted noise gradually; it does not reveal a hidden clean image in one step.",
  transformer:"A high attention weight shows routing inside the model, not a complete causal explanation of its answer.",
  llm:"Temperature changes how token probabilities are sampled; it does not change the model's learned knowledge.",
  vlm:"Image–text similarity measures alignment in a representation space, not guaranteed factual understanding."
};
const backpropStages=[
  {label:"Before calculating",title:"Predict the update",prompt:"If the target is above the current prediction, should the next update push the prediction up, down, or leave it similar? Choose before running the graph."},
  {label:"Run the mechanism",title:"Follow the forward pass",prompt:"Select “Forward pass.” Trace x₁ and x₂ through multiplication, addition, activation, and loss. Change only one weight and run it again."},
  {label:"Look for evidence",title:"Trace responsibility backward",prompt:"Select “Backward pass.” Compare the two weight gradients. Which input makes its connected weight more sensitive? Then try the vanishing-gradient preset."},
  {label:"Connect picture and rule",title:"Explain one update",prompt:"Apply the update and inspect the loss. Explain why each parameter moves opposite its gradient, and why saturation can nearly stop learning."}
];

let learningState={item:learningItems[0],mode:"guided",stage:0,prediction:"",restoring:false};
const learningShell=$("#learningShell"),guidedPanel=$("#guidedPanel"),guideTitle=$("#guideTitle"),guidePrompt=$("#guidePrompt"),guideStageLabel=$("#guideStageLabel"),predictionOptions=$("#predictionOptions"),misconceptionNote=$("#misconceptionNote");

function safePreference(key,value){try{if(value===undefined)return localStorage.getItem(key);localStorage.setItem(key,value)}catch(error){return null}}
function genericStages(item){return [
  {label:"Before changing anything",title:"Make a prediction",prompt:`Choose increase, decrease, or stay similar for the main result. Then decide which control in ${item.title} you expect to have the strongest effect.`},
  {label:"Change one thing",title:"Run a small experiment",prompt:item.interaction+" Keep the other controls fixed so the cause of the change stays visible."},
  {label:"Look for evidence",title:"Observe more than one view",prompt:`Use ${item.visual.toLowerCase()} Compare the picture with at least one numerical readout.`},
  {label:"Connect interaction and idea",title:"Explain what changed",prompt:`Complete this sentence: “When I changed ___, ___ changed because ___.” Use the Intuition and Live math tabs if you need another clue.`}
]}
function currentStages(){return learningState.item.id==="backprop"?backpropStages:genericStages(learningState.item)}
function setMode(mode,save=true){learningState.mode=mode;const guided=mode==="guided";learningShell.classList.toggle("explore-mode",!guided);guidedPanel.hidden=!guided;$("#guidedModeButton").classList.toggle("active",guided);$("#exploreModeButton").classList.toggle("active",!guided);$("#guidedModeButton").setAttribute("aria-pressed",String(guided));$("#exploreModeButton").setAttribute("aria-pressed",String(!guided));$("#learningShellTitle").textContent=guided?"Guided exploration":"Free exploration";if(save)safePreference("ml-lab-mode",mode);renderGuide();updateLessonAddress()}
function setStage(stage){learningState.stage=clamp(stage,0,3);renderGuide();updateLessonAddress()}
function renderGuide(){const stage=currentStages()[learningState.stage],isPrediction=learningState.stage===0;guideStageLabel.textContent=stage.label;guideTitle.textContent=stage.title;guidePrompt.textContent=stage.prompt;predictionOptions.hidden=!isPrediction;document.querySelectorAll(".guide-step").forEach((button,index)=>{const active=index===learningState.stage;button.classList.toggle("active",active);button.setAttribute("aria-current",active?"step":"false")});document.querySelectorAll("[data-prediction]").forEach(button=>button.classList.toggle("selected",button.dataset.prediction===learningState.prediction));misconceptionNote.innerHTML=`<strong>Check a misconception</strong><p>${misconceptionPrompts[learningState.item.id]||"One attractive visual result is not enough evidence; compare the display, the numbers, and the underlying rule."}</p>`;$("#previousStageButton").disabled=learningState.stage===0;$("#nextStageButton").textContent=learningState.stage===3?"Return to prediction":"Continue";const itemIndex=learningItems.indexOf(learningState.item);$("#previousLessonButton").disabled=itemIndex===0;$("#nextLessonButton").disabled=itemIndex===learningItems.length-1}
function activeSettings(){const active=document.querySelector(".module.active");if(!active)return{};const values={};active.querySelectorAll("input[id], select[id]").forEach(control=>{if(control.type==="radio"&&!control.checked)return;if(control.type==="checkbox")values[control.id]=control.checked?"1":"0";else values[control.id]=control.value});return values}
function updateLessonAddress(){if(learningState.restoring)return;const params=new URLSearchParams();params.set("module",learningState.item.id);params.set("mode",learningState.mode);params.set("stage",String(learningState.stage));const settings=activeSettings();if(Object.keys(settings).length)params.set("settings",JSON.stringify(settings));history.replaceState(null,"",`${location.pathname}${location.search}#${params.toString()}`)}
function restoreSettings(values){Object.entries(values||{}).forEach(([id,value])=>{const control=document.getElementById(id);if(!control)return;if(control.type==="checkbox"){control.checked=value==="1";control.dispatchEvent(new Event("change",{bubbles:true}))}else if(control.type==="radio"){control.checked=control.value===value;if(control.checked)control.dispatchEvent(new Event("change",{bubbles:true}))}else{control.value=value;control.dispatchEvent(new Event(control.tagName==="SELECT"?"change":"input",{bubbles:true}))}})}
function moveLesson(amount){const index=learningItems.indexOf(learningState.item),next=learningItems[index+amount];if(!next)return;learningState.stage=0;learningState.prediction="";openModule(next);learningShell.scrollIntoView({behavior:document.body.classList.contains("reduce-motion")?"auto":"smooth",block:"start"})}

window.updateLearningGuide=item=>{learningState.item=item;learningState.stage=0;learningState.prediction="";renderGuide();updateLessonAddress();requestAnimationFrame(()=>requestAnimationFrame(updateLessonAddress))};
$("#guidedModeButton").addEventListener("click",()=>setMode("guided"));$("#exploreModeButton").addEventListener("click",()=>setMode("explore"));document.querySelectorAll(".guide-step").forEach(button=>button.addEventListener("click",()=>setStage(Number(button.dataset.guideStage))));document.querySelectorAll("[data-prediction]").forEach(button=>button.addEventListener("click",()=>{learningState.prediction=button.dataset.prediction;renderGuide();updateLessonAddress()}));$("#previousStageButton").addEventListener("click",()=>setStage(learningState.stage-1));$("#nextStageButton").addEventListener("click",()=>setStage(learningState.stage===3?0:learningState.stage+1));$("#previousLessonButton").addEventListener("click",()=>moveLesson(-1));$("#nextLessonButton").addEventListener("click",()=>moveLesson(1));
$("#animationSpeed").addEventListener("change",event=>{window.labAnimationSpeed=Number(event.target.value);safePreference("ml-lab-speed",event.target.value);updateLessonAddress()});$("#motionButton").addEventListener("click",()=>{const reduced=!document.body.classList.contains("reduce-motion");document.body.classList.toggle("reduce-motion",reduced);$("#motionButton").classList.toggle("active",reduced);$("#motionButton").setAttribute("aria-pressed",String(reduced));safePreference("ml-lab-motion",reduced?"1":"0");updateLessonAddress()});
document.addEventListener("input",event=>{if(event.target.closest(".module"))clearTimeout(window.learningAddressTimer),window.learningAddressTimer=setTimeout(updateLessonAddress,180)});document.addEventListener("change",event=>{if(event.target.closest(".module"))updateLessonAddress()});
document.addEventListener("click",event=>{if(event.target.closest(".module button"))setTimeout(updateLessonAddress,0)});
$("#shareLessonButton").addEventListener("click",async()=>{updateLessonAddress();const button=$("#shareLessonButton"),original=button.textContent;try{await navigator.clipboard.writeText(location.href);button.textContent="Link copied"}catch(error){const field=document.createElement("textarea");field.value=location.href;document.body.appendChild(field);field.select();document.execCommand("copy");field.remove();button.textContent="Link copied"}setTimeout(()=>button.textContent=original,1400)});

(function initialiseLearning(){const params=new URLSearchParams(location.hash.slice(1)),requested=learningItems.find(item=>item.id===params.get("module"))||learningItems[0],requestedStage=clamp(Number(params.get("stage")||0),0,3),storedMode=safePreference("ml-lab-mode"),mode=params.get("mode")||storedMode||"guided",speed=safePreference("ml-lab-speed")||"1",reduced=safePreference("ml-lab-motion")==="1";learningState.restoring=true;learningState.item=requested;learningState.stage=requestedStage;$("#animationSpeed").value=speed;window.labAnimationSpeed=Number(speed);document.body.classList.toggle("reduce-motion",reduced);$("#motionButton").classList.toggle("active",reduced);$("#motionButton").setAttribute("aria-pressed",String(reduced));setMode(mode,false);openModule(requested);let settings={};try{settings=JSON.parse(params.get("settings")||"{}")||{}}catch(error){settings={}}requestAnimationFrame(()=>requestAnimationFrame(()=>{learningState.stage=requestedStage;restoreSettings(settings);learningState.restoring=false;renderGuide();updateLessonAddress()}))})();
