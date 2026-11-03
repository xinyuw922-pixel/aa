"use strict";

const $ = id => document.getElementById(id);
const names = {doorkey:"DoorKey",keycorridor:"KeyCorridor",lavagap:"LavaGap",memory:"Memory",multiroom:"MultiRoom",redblue:"RedBlueDoor",unlockpickup:"UnlockPickup",blockedunlockpickup:"BlockedUnlockPickup"};
const variants = {full:"Full",nocue:"NoCue",cf:"CF"};
const styles = {clean:"Original",noisy:"Noise",noise:"Noise",style:"Style"};
const taskNames = {A:"Action-conditioned transition",B:"Single-frame observation",C:"Goal-conditioned trajectory",D:"Reference-conditioned judgment",E:"Prefix state tracking"};
let data, current, families = [];
const state = {task:"C",env:"keycorridor",case:"",variant:"full",style:"clean",rep:"composite",revealed:false};

function icons(){window.lucide?.createIcons();}
function element(tag, className, text){const el=document.createElement(tag);if(className)el.className=className;if(text!==undefined)el.textContent=text;return el;}
function unique(items){return [...new Set(items)];}
function options(select, values, label, preferred){select.replaceChildren();for(const value of values){const option=element("option","",label(value));option.value=value;select.append(option);}select.value=values.includes(preferred)?preferred:values[0];select.disabled=values.length<2;return select.value;}
function openImage(src,title){$("dialog-image").src=src;$("dialog-title").textContent=title;$("image-dialog").showModal();}
function original(asset,label){const fig=element("figure","original-figure");const img=element("img");img.src=asset.src;img.alt=label;img.loading="lazy";img.addEventListener("click",()=>openImage(asset.src,label));fig.append(img);return fig;}
function crop(asset,box,label){const viewport=element("div","crop");viewport.style.aspectRatio=box.width+" / "+box.height;const img=element("img");img.src=asset.src;img.alt=label;img.loading="lazy";img.style.width=100*asset.width/box.width+"%";img.style.left=-100*box.x/box.width+"%";img.style.top=-100*box.y/box.height+"%";viewport.append(img);viewport.addEventListener("click",()=>openImage(asset.src,label));return viewport;}
function panels(asset, boxes){const grid=element("div","split-grid");for(const box of boxes){const fig=element("figure");fig.append(crop(asset,box,box.label),element("figcaption","",box.label));grid.append(fig);}return grid;}
function syncHash(){const query=new URLSearchParams({task:state.task,env:state.env,case:state.case,variant:state.variant,style:state.style,rep:state.rep});history.replaceState(null,"","#"+query);}

function render(){
  const taskRows=data.examples.filter(r=>r.task===state.task);
  state.env=options($("environment"),unique(taskRows.map(r=>r.env)),value=>names[value]||value,state.env);
  const envRows=taskRows.filter(r=>r.env===state.env);
  families=unique(envRows.map(r=>r.case_id));
  state.case=options($("case"),families,value=>{
    const index=families.indexOf(value)+1;
    return "Example "+String(index).padStart(2,"0")+" · "+value.split("/").pop().split(".").slice(-1)[0];
  },state.case);
  const caseRows=envRows.filter(r=>r.case_id===state.case);
  state.variant=options($("variant"),unique(caseRows.map(r=>r.variant)).sort((a,b)=>["full","nocue","cf"].indexOf(a)-["full","nocue","cf"].indexOf(b)),v=>variants[v]||v,state.variant);
  const variantRows=caseRows.filter(r=>r.variant===state.variant);
  state.style=options($("style"),unique(variantRows.map(r=>r.style)),v=>styles[v]||v,state.style);
  current=variantRows.find(r=>r.style===state.style);
  if(!current)throw Error("No exact example matching the selected dimensions");
  const supportsSplit=!!(current.split.length||current.panels.length||current.reference);
  if(!supportsSplit)state.rep="composite";
  $("split").disabled=!supportsSplit;
  $("split").title=supportsSplit?"Same evidence, separate panels":"No paired Split view is available for this exam";
  $("composite").setAttribute("aria-pressed",state.rep==="composite");
  $("split").setAttribute("aria-pressed",state.rep==="split");
  document.querySelectorAll("[data-task]").forEach(button=>{button.setAttribute("aria-selected",button.dataset.task===state.task);button.tabIndex=button.dataset.task===state.task?0:-1;button.title=taskNames[button.dataset.task];});
  $("evidence-label").textContent=taskNames[state.task];
  const evidence=$("evidence");evidence.replaceChildren();
  if(current.task==="D"){
    const paired=element("div","paired-evidence");
    for(const key of ["reference","query"]){const box=current[key];const fig=element("figure","original-figure");const label=element("figcaption","figure-label",box.label);label.append(element("span","known-success",key==="reference"?"Known success":"Outcome to judge"));fig.append(label,state.rep==="split"?panels(current.composite,box.panels):crop(current.composite,box,box.label));paired.append(fig);}
    evidence.append(paired);
  }else if(state.rep==="split"&&current.split.length){
    const grid=element("div","split-grid action");
    for(const asset of current.split){const fig=original(asset,asset.label);fig.append(element("figcaption","",asset.label));grid.append(fig);}evidence.append(grid);
  }else if(state.rep==="split"){
    evidence.append(panels(current.composite,current.panels));
  }else{
    evidence.append(original(current.composite,"Task "+current.task+" · "+(names[current.env]||current.env)));
  }
  $("prompt").textContent=current.prompt;
  $("question-label").textContent=current.action?"Action: "+current.action:taskNames[state.task];
  $("answer").textContent=typeof current.answer==="object"?JSON.stringify(current.answer,null,2):String(current.answer);
  $("answer").hidden=!state.revealed;
  $("reveal").hidden=state.revealed;
  $("answer-status").textContent=state.revealed?(current.task==="D"?"Query label · reference is a known success":"Recorded answer key"):"";
  $("sample-counter").textContent=String(families.indexOf(state.case)+1).padStart(2,"0")+" / "+String(families.length).padStart(2,"0");
  $("source-uid").textContent=current.uid;
  $("original-image").href=current.composite.src;
  $("explorer").setAttribute("aria-busy","false");
  syncHash();icons();
}

function showcase(){
  const target=$("triplet");target.replaceChildren();
  const descriptions={full:"Original evidence",nocue:"Evidence removed · outcome preserved",cf:"Causal intervention · outcome changed"};
  for(const [index,variant] of ["full","nocue","cf"].entries()){
    const row=data.examples.find(r=>r.uid===data.showcase[variant]);
    const article=element("article","triplet-item");const head=element("div","triplet-header");const title=element("span","triplet-title");title.append(element("span","triplet-index","0"+(index+1)),document.createTextNode(variants[variant]));
    const verdict=element("span","verdict"+(variant==="cf"?" fail":""));const icon=element("i");icon.setAttribute("data-lucide",variant==="cf"?"x":"check");verdict.append(icon,document.createTextNode(row.answer));head.append(title,verdict);
    const fig=element("figure");const img=element("img","triplet-image");img.src=row.composite.src;img.alt=variants[variant]+" evidence from "+row.uid;img.addEventListener("click",()=>openImage(img.src,row.uid));fig.append(img,element("figcaption","",descriptions[variant]));article.append(head,fig);target.append(article);
    $("showcase-source").textContent="Task C · "+names[row.env]+" · "+row.group;
  }
  icons();
}

for(const name of ["environment","case","variant","style"]){$(name).addEventListener("change",event=>{const key=name==="environment"?"env":name;state[key]=event.target.value;if(key==="env")state.case="";state.revealed=false;render();});}
document.querySelectorAll("[data-task]").forEach(button=>{
  button.addEventListener("click",()=>{state.task=button.dataset.task;state.case="";state.variant="full";state.style="clean";state.revealed=false;render();});
  button.addEventListener("keydown",event=>{if(!["ArrowLeft","ArrowRight"].includes(event.key))return;event.preventDefault();const tabs=[...document.querySelectorAll("[data-task]")];const index=tabs.indexOf(button);const next=tabs[(index+(event.key==="ArrowRight"?1:-1)+tabs.length)%tabs.length];next.click();next.focus();});
});
for(const rep of ["composite","split"]){$(rep).addEventListener("click",()=>{state.rep=rep;render();});}
for(const [id,delta] of [["previous",-1],["next",1]]){$(id).addEventListener("click",()=>{state.case=families[(families.indexOf(state.case)+delta+families.length)%families.length];state.revealed=false;render();});}
$("reveal").addEventListener("click",()=>{state.revealed=true;render();});
$("close-dialog").addEventListener("click",()=>$("image-dialog").close());
$("image-dialog").addEventListener("click",event=>{if(event.target===$("image-dialog"))$("image-dialog").close();});
$("share").addEventListener("click",async()=>{try{await navigator.clipboard.writeText(location.href);$("toast").textContent="Example link copied";}catch{$("toast").textContent=location.href;}$("toast").hidden=false;setTimeout(()=>$("toast").hidden=true,2500);});
fetch("examples.json").then(response=>{if(!response.ok)throw Error("Example data unavailable");return response.json();}).then(value=>{
  data=value;const query=new URLSearchParams(location.hash.slice(1));for(const key of ["task","env","case","variant","style","rep"]){if(query.has(key))state[key]=query.get(key);}
  if(!"ABCDE".includes(state.task)||state.task.length!==1)state.task="C";
  if(!["composite","split"].includes(state.rep))state.rep="composite";
  showcase();render();
}).catch(error=>{$("evidence").textContent="Examples could not be loaded.";$("triplet").textContent="Examples unavailable";console.error(error);});
icons();
