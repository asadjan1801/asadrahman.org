"use strict";

/* Optimizer update rules for the Gradient Descent module, on a 2D loss surface. Pure functions; the Node tests
   import this file. State: position θ, velocity v (momentum / Nesterov, also the last step taken), first moment m,
   second moment s and step count t. Every position is clipped to the plotted square [−bound, bound]². */
const OPTIM_NAMES={momentum:"Momentum",nesterov:"Nesterov",rmsprop:"RMSProp",adam:"Adam"};
function optimInit(position){return {position:{x:position.x,y:position.y},velocity:{x:0,y:0},m:{x:0,y:0},s:{x:0,y:0},t:0}}
function optimClip(value,bound){return Math.max(-bound,Math.min(bound,value))}
/* One step. gradient(x, y) returns {x, y}. Hyperparameters: rate α, momentum β (momentum and Nesterov),
   rho ρ = 0.9 (RMSProp), beta1 = 0.9 and beta2 = 0.999 (Adam), epsilon ε = 1e-8.
     momentum (heavy ball): v ← βv − α∇L(θ);            θ ← θ + v          (β = 0 is plain gradient descent)
     nesterov:              v ← βv − α∇L(θ + βv);       θ ← θ + v
     rmsprop:               s ← ρs + (1 − ρ)g²;         θ ← θ − α g / (√s + ε)
     adam:                  m ← β₁m + (1 − β₁)g;  s ← β₂s + (1 − β₂)g²;  m̂ = m/(1 − β₁ᵗ), ŝ = s/(1 − β₂ᵗ);  θ ← θ − α m̂ / (√ŝ + ε) */
function optimStep(name,state,gradient,{rate,momentum=0,rho=.9,beta1=.9,beta2=.999,epsilon=1e-8,bound=5}){
  const p=state.position,v=state.velocity,next={position:{...p},velocity:{...v},m:{...state.m},s:{...state.s},t:state.t+1};let g;
  if(name==="momentum"||name==="nesterov"){
    g=name==="nesterov"?gradient(p.x+momentum*v.x,p.y+momentum*v.y):gradient(p.x,p.y);
    next.velocity={x:momentum*v.x-rate*g.x,y:momentum*v.y-rate*g.y};
  }else if(name==="rmsprop"){
    g=gradient(p.x,p.y);next.s={x:rho*state.s.x+(1-rho)*g.x*g.x,y:rho*state.s.y+(1-rho)*g.y*g.y};
    next.velocity={x:-rate*g.x/(Math.sqrt(next.s.x)+epsilon),y:-rate*g.y/(Math.sqrt(next.s.y)+epsilon)};
  }else{
    g=gradient(p.x,p.y);next.m={x:beta1*state.m.x+(1-beta1)*g.x,y:beta1*state.m.y+(1-beta1)*g.y};next.s={x:beta2*state.s.x+(1-beta2)*g.x*g.x,y:beta2*state.s.y+(1-beta2)*g.y*g.y};
    const c1=1-beta1**next.t,c2=1-beta2**next.t;next.velocity={x:-rate*(next.m.x/c1)/(Math.sqrt(next.s.x/c2)+epsilon),y:-rate*(next.m.y/c1)/(Math.sqrt(next.s.y/c2)+epsilon)};
  }
  next.position={x:optimClip(p.x+next.velocity.x,bound),y:optimClip(p.y+next.velocity.y,bound)};next.gradient=g;
  return next;
}
/* For a quadratic with Hessian eigenvalues λ_min ≤ λ_max: condition number κ = λ_max/λ_min, and the momentum β that
   critically damps the slow direction at step size α, β* = (1 − √(αλ_min))² (Goh, “Why Momentum Really Works”, 2017). */
function optimCriticalMomentum(rate,lambdaMin){return (1-Math.sqrt(Math.max(0,rate*lambdaMin)))**2}

const OPTIM_MATH={OPTIM_NAMES,optimInit,optimStep,optimCriticalMomentum};
if(typeof module!=="undefined"&&module.exports)module.exports=OPTIM_MATH;
