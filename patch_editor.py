#!/usr/bin/env python3
"""Run this once to patch your HTML file"""
import re, os

path = os.path.join(os.path.dirname(__file__), "ai-image-editor-pro.html")
with open(path, "r", encoding="utf-8") as f:
    code = f.read()

# Check if already patched
if "removeBgLocal" in code:
    print("✅ Already patched! Restart rmbg_server.py and open http://localhost:8080")
    exit()

# PATCH 1: Remove all HuggingFace imports
code = re.sub(
    r"import\s*\{[^}]*\}\s*from\s*['"]https://cdn\.jsdelivr\.net/npm/@huggingface[^'"]*['"];?",
    "// HuggingFace removed — using local rembg server",
    code
)

# PATCH 2: Replace loadBgModel + removeBackground section
old_marker = "async function loadBgModel(){"
new_bg = """
/* ── LOCAL REMBG SERVER (python3 rmbg_server.py) ── */
function buildCropSrc(){
  const w=appState.cropW??originalImage.width;
  const h=appState.cropH??originalImage.height;
  const off=document.createElement('canvas');
  off.width=w;off.height=h;
  off.getContext('2d').drawImage(originalImage,appState.cropX,appState.cropY,w,h,0,0,w,h);
  return off;
}
async function checkLocalServer(){
  try{
    const r=await fetch('http://localhost:8080/health',{signal:AbortSignal.timeout(1500)});
    const d=await r.json();
    return d.rembg===true;
  }catch(e){return false;}
}
async function removeBgLocal(srcCanvas){
  const res=await fetch('http://localhost:8080/removebg',{
    method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({image:srcCanvas.toDataURL('image/png')})
  });
  if(!res.ok) throw new Error('Server '+res.status);
  const d=await res.json();
  if(!d.success) throw new Error(d.error||'rembg failed');
  const bytes=Uint8Array.from(atob(d.image.split(',')[1]),c=>c.charCodeAt(0));
  return await createImageBitmap(new Blob([bytes],{type:'image/png'}));
}
function colDist(r1,g1,b1,r2,g2,b2){
  const rm=(r1+r2)/2,dr=r1-r2,dg=g1-g2,db=b1-b2;
  return Math.sqrt((2+rm/256)*dr*dr+4*dg*dg+(2+(255-rm)/256)*db*db);
}
function kmeans2(s,it=10){
  let c1=s[0],c2=s[Math.floor(s.length*.6)];
  for(let i=0;i<it;i++){
    let a=[0,0,0,0],b=[0,0,0,0];
    for(const [r,g,bl] of s){
      const d1=colDist(r,g,bl,c1[0],c1[1],c1[2]),d2=colDist(r,g,bl,c2[0],c2[1],c2[2]);
      if(d1<=d2){a[0]+=r;a[1]+=g;a[2]+=bl;a[3]++;}
      else{b[0]+=r;b[1]+=g;b[2]+=bl;b[3]++;}
    }
    if(a[3])c1=[a[0]/a[3],a[1]/a[3],a[2]/a[3]];
    if(b[3])c2=[b[0]/b[3],b[1]/b[3],b[2]/b[3]];
  }
  return[c1,c2];
}
function gaussBlur(data,w,h,r){
  const k=[],s=r/2,s2=2*s*s;let sum=0;
  for(let i=-r;i<=r;i++){const v=Math.exp(-(i*i)/s2);k.push(v);sum+=v;}
  k.forEach((_,i)=>k[i]/=sum);
  const t=new Float32Array(w*h),o=new Float32Array(w*h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let a=0;for(let i=-r;i<=r;i++)a+=data[y*w+Math.min(Math.max(x+i,0),w-1)]*k[i+r];
    t[y*w+x]=a;
  }
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
    let a=0;for(let i=-r;i<=r;i++)a+=t[Math.min(Math.max(y+i,0),h-1)*w+x]*k[i+r];
    o[y*w+x]=a;
  }
  return o;
}
function floodFill(px,w,h,c1,c2,thr){
  const mask=new Uint8Array(w*h),q=[];
  function isBg(i){const r=px[i*4],g=px[i*4+1],b=px[i*4+2];return colDist(r,g,b,c1[0],c1[1],c1[2])<=thr||colDist(r,g,b,c2[0],c2[1],c2[2])<=thr;}
  function enq(x,y){const i=y*w+x;if(!mask[i]&&isBg(i)){mask[i]=1;q.push(i);}}
  for(let x=0;x<w;x++){enq(x,0);enq(x,h-1);}
  for(let y=0;y<h;y++){enq(0,y);enq(w-1,y);}
  let head=0;
  while(head<q.length){const i=q[head++],x=i%w,y=Math.floor(i/w);if(x>0)enq(x-1,y);if(x<w-1)enq(x+1,y);if(y>0)enq(x,y-1);if(y<h-1)enq(x,y+1);}
  return mask;
}
function removeBgCanvas(src){
  const w=src.width,h=src.height,px=src.getContext('2d').getImageData(0,0,w,h).data;
  const B=Math.max(3,Math.round(Math.min(w,h)*.05)),samp=[];
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(x<B||x>=w-B||y<B||y>=h-B){const i=(y*w+x)*4;if(px[i+3]>10)samp.push([px[i],px[i+1],px[i+2]]);}
  const[c1,c2]=kmeans2(samp.length?samp:[[255,255,255]]);
  let v=0;for(const[r,g,b] of samp.slice(0,500))v+=colDist(r,g,b,c1[0],c1[1],c1[2]);
  v/=Math.max(1,Math.min(samp.length,500));
  const thr=Math.min(Math.max(v*2.5+18,22),75);
  const mask=floodFill(px,w,h,c1,c2,thr),alpha=new Float32Array(w*h);
  for(let i=0;i<w*h;i++)alpha[i]=mask[i]?0:1;
  const bl=gaussBlur(alpha,w,h,Math.max(2,Math.round(Math.min(w,h)*.008)));
  const out=document.createElement('canvas');out.width=w;out.height=h;
  const oc=out.getContext('2d');oc.drawImage(src,0,0);
  const od=oc.getImageData(0,0,w,h);
  for(let i=0;i<w*h;i++){const v2=bl[i];od.data[i*4+3]=v2<.15?0:v2>.85?255:Math.round(((v2-.15)/.7)*255);}
  oc.putImageData(od,0,0);return out;
}
async function removeBackground(token){
  if(isRemovingBg||!originalImage)return;
  isRemovingBg=true;rmbgOverlay.classList.add('active');
  setStatus('Removing background…','working');
  try{
    await new Promise(r=>setTimeout(r,30));
    if(imageToken!==token){isRemovingBg=false;rmbgOverlay.classList.remove('active');return;}
    const cropSrc=buildCropSrc();
    let result=null;
    const up=await checkLocalServer().catch(()=>false);
    if(up){
      try{
        rmbgMsg.textContent='Running rembg AI…';
        rmbgMsg.nextElementSibling.textContent='U2Net — professional quality';
        const bmp=await removeBgLocal(cropSrc);
        const c=document.createElement('canvas');c.width=bmp.width;c.height=bmp.height;
        c.getContext('2d').drawImage(bmp,0,0);result=c;
        toast('Background removed! ✨ U2Net AI');
      }catch(e){console.warn('rembg:',e);toast('rembg error, using canvas…',2000);}
    }
    if(!result){
      rmbgMsg.textContent='Canvas mode…';
      rmbgMsg.nextElementSibling.textContent='Start rmbg_server.py for AI quality';
      await new Promise(r=>setTimeout(r,20));
      result=removeBgCanvas(cropSrc);
      toast('Done! Run rmbg_server.py for AI quality');
    }
    if(imageToken!==token)return;
    maskedCanvas=result;
    mutate(()=>{appState.hasMask=true;},'Remove BG');
    setStatus('Background removed!','ready');
  }catch(err){
    console.error(err);setStatus('Failed: '+err.message,'error');toast('Failed: '+err.message);maskedCanvas=null;
  }finally{
    isRemovingBg=false;rmbgOverlay.classList.remove('active');
  }
}
"""

# Find and replace the entire old BG section
idx_start = code.find("async function loadBgModel(){")
# find end = start of crop section
idx_end = code.find("/* ── CROP ── */")
if idx_start == -1 or idx_end == -1:
    print(f"❌ Markers not found: loadBgModel={idx_start}, CROP={idx_end}")
    # Try alternate
    idx_start = code.find("let bgModel")
    print(f"   Trying 'let bgModel' at {idx_start}")
else:
    code = code[:idx_start] + new_bg + "\n" + code[idx_end:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    print("✅ File patched successfully!")
    print("✅ No more HuggingFace/SegformerFor errors")
    print("👉 Now run: python3 rmbg_server.py")
    print("👉 Open:    http://localhost:8080")
