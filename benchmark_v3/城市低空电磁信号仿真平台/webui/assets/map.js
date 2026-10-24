import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const colors={communication:0x367bf0,interference:0xe98942,background:0x8a78cd,receiver:0x129e87};
// 与后端 cityrf.radiomap.RAMP_STOPS 保持一致，保证网页热力图与导出 PNG 同色。
export const RAMP_STOPS=[[0.19,0.07,0.23],[0.27,0.31,0.71],[0.10,0.60,0.79],[0.10,0.84,0.60],[0.55,0.94,0.27],[0.93,0.79,0.20],[0.97,0.50,0.13],[0.80,0.16,0.08],[0.48,0.02,0.01]];
export function rampColor(t){
 const x=Math.max(0,Math.min(1,t))*(RAMP_STOPS.length-1);
 const i=Math.min(Math.floor(x),RAMP_STOPS.length-2),f=x-i,a=RAMP_STOPS[i],b=RAMP_STOPS[i+1];
 return [Math.round((a[0]+(b[0]-a[0])*f)*255),Math.round((a[1]+(b[1]-a[1])*f)*255),Math.round((a[2]+(b[2]-a[2])*f)*255)];
}
export function rampCss(t){const [r,g,b]=rampColor(t);return `rgb(${r},${g},${b})`;}
export class SceneMap{
 constructor(container,onSelect,onMove){
  this.el=container;this.onSelect=onSelect;this.onMove=onMove;this.mode='3d';this.items=[];this.showLinks=false;this.showLabels=true;this.radioMap=null;this.radioMapData=null;this.radioMapMode='ghost';
  this.scene=new THREE.Scene();this.scene.background=new THREE.Color('#edf1f6');
  this.renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});this.renderer.setPixelRatio(Math.min(devicePixelRatio,2));container.append(this.renderer.domElement);
  this.renderer.outputColorSpace=THREE.SRGBColorSpace;
  this.perspective=new THREE.PerspectiveCamera(40,1,.1,6000);this.perspective.up.set(0,0,1);
  this.orthographic=new THREE.OrthographicCamera(-400,400,300,-300,.1,6000);this.orthographic.up.set(0,1,0);
  this.camera=this.perspective;
  this.scene.add(new THREE.HemisphereLight(0xffffff,0x9aafc6,2.6));
  const light=new THREE.DirectionalLight(0xffffff,2.3);light.position.set(-250,-100,500);this.scene.add(light);
  this.grid=new THREE.GridHelper(2000,80,0xb9c9dc,0xd5dee9);this.grid.rotation.x=Math.PI/2;this.grid.position.z=.15;this.scene.add(this.grid);
  this.ground=new THREE.Mesh(new THREE.PlaneGeometry(2200,1900),new THREE.MeshStandardMaterial({color:0xe9eef4,roughness:1}));this.ground.position.z=-.2;this.scene.add(this.ground);
  this.devices=new THREE.Group();this.links=new THREE.Group();this.scene.add(this.devices,this.links);
  this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.target.set(0,0,15);this.controls.maxDistance=2800;this.controls.maxPolarAngle=Math.PI/2-.03;
  this.reset();this.ray=new THREE.Raycaster();this.pointer=new THREE.Vector2();this.drag=null;
  const canvas=this.renderer.domElement;
  canvas.addEventListener('pointerdown',e=>this.down(e),{capture:true});
  canvas.addEventListener('pointermove',e=>this.move(e));
  canvas.addEventListener('pointerup',e=>this.up(e));canvas.addEventListener('pointercancel',()=>{this.drag=null;this.controls.enabled=true});
  new ResizeObserver(()=>this.resize()).observe(container);this.resize();
  this.renderer.setAnimationLoop(()=>{this.controls.update();for(const o of this.devices.children){if(o.isSprite){const unit=this.camera.isPerspectiveCamera?2*this.camera.position.distanceTo(o.position)*Math.tan(this.camera.fov*Math.PI/360)/this.el.clientHeight:(this.camera.top-this.camera.bottom)/this.camera.zoom/this.el.clientHeight;o.scale.set(110*unit,24*unit,1);}}this.renderer.render(this.scene,this.camera)});
 }
 async load(meshId='munich'){
  this.clearRadioMap();
  const [meta,buffer]=await Promise.all([fetch(`/assets/${meshId}_mesh.json`).then(r=>r.json()),fetch(`/assets/${meshId}_mesh.bin`).then(r=>r.arrayBuffer())]);
  const raw=new Float32Array(buffer),filtered=[];
  // Preserve all non-ground triangles, using the original world coordinates.
  for(let i=0;i<raw.length;i+=9){if(Math.max(raw[i+2],raw[i+5],raw[i+8])>.5)for(let j=0;j<9;j++)filtered.push(raw[i+j]);}
  if(this.buildings){this.scene.remove(this.buildings,this.edges);this.buildings.geometry.dispose();this.edges.geometry.dispose();}
  const geo=new THREE.BufferGeometry();geo.setAttribute('position',new THREE.Float32BufferAttribute(filtered,3));geo.computeVertexNormals();
  this.buildings=new THREE.Mesh(geo,new THREE.MeshStandardMaterial({color:0xd0d9e5,roughness:.9,side:THREE.DoubleSide}));this.scene.add(this.buildings);
  this.edges=new THREE.LineSegments(new THREE.EdgesGeometry(geo,35),new THREE.LineBasicMaterial({color:0xa9b9cc,transparent:true,opacity:.3}));this.scene.add(this.edges);
  // 按 AOI 范围自适应地面/网格线/相机（慕尼黑约 1.5 km，自定义 AOI 可小可大）
  const raw_b=meta.bounds||[[-750,-950,0],[750,950,0]];
  const b=Array.isArray(raw_b)?{min:raw_b[0],max:raw_b[1]}:raw_b;
  const span=Math.max(b.max[0]-b.min[0],b.max[1]-b.min[1],200);this.span=span;
  this.scene.remove(this.grid,this.ground);this.grid.geometry.dispose();this.ground.geometry.dispose();
  this.grid=new THREE.GridHelper(Math.ceil(span*2/100)*100,80,0xb9c9dc,0xd5dee9);this.grid.rotation.x=Math.PI/2;this.grid.position.z=.15;this.scene.add(this.grid);
  this.ground=new THREE.Mesh(new THREE.PlaneGeometry(span*3,span*3),new THREE.MeshStandardMaterial({color:0xe9eef4,roughness:1}));this.ground.position.z=-.2;this.scene.add(this.ground);
  this.controls.maxDistance=span*4;this.perspective.far=Math.max(6000,span*6);this.perspective.updateProjectionMatrix();
  document.querySelector('#mesh-tag').textContent=`${meta.name||meshId} · ${(meta.triangle_count/1000).toFixed(1)}k 三角面`;
  this.reset();this.resize();
 }
 resize(){const w=this.el.clientWidth,h=this.el.clientHeight;if(!w||!h)return;this.renderer.setSize(w,h);this.perspective.aspect=w/h;this.perspective.updateProjectionMatrix();const span=(this.span||1500)*.2;this.orthographic.left=-span*w/h;this.orthographic.right=span*w/h;this.orthographic.top=span;this.orthographic.bottom=-span;this.orthographic.updateProjectionMatrix();}
 reset(){
  const objects=this.config?[...this.config.emitters,...this.config.receivers]:[];
  const c=new THREE.Vector3();if(objects.length){for(const d of objects)c.add(new THREE.Vector3(...d.position_m));c.divideScalar(objects.length);c.z=15;}
  this.controls.target.copy(c);
  let extent=this.span?this.span*.55:230;for(const d of objects)extent=Math.max(extent,Math.hypot(d.position_m[0]-c.x,d.position_m[1]-c.y)*1.15);
  this.perspective.position.set(c.x+extent*.7,c.y-extent*1.35,c.z+extent*1.45);
  this.orthographic.position.set(c.x,c.y,1500);this.orthographic.zoom=Math.min(1.15,270/extent);this.orthographic.updateProjectionMatrix();this.controls.update();
 }
 setMode(mode){this.mode=mode;this.camera=mode==='2d'?this.orthographic:this.perspective;this.controls.dispose();this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.enableRotate=mode==='3d';this.controls.maxDistance=(this.span||1500)*4;this.controls.maxPolarAngle=mode==='3d'?Math.PI/2-.03:Math.PI;this.reset();this.resize();}
 projectPointer(e){const r=this.el.getBoundingClientRect();this.pointer.set((e.clientX-r.left)/r.width*2-1,-(e.clientY-r.top)/r.height*2+1);this.ray.setFromCamera(this.pointer,this.camera);}
 down(e){
  if(e.button!==0)return;this.projectPointer(e);const hit=this.ray.intersectObjects(this.items,false)[0];
  if(!hit)return;const d=hit.object.userData;this.onSelect(d.type,d.index);
  if(this.mode==='2d'){
   e.stopImmediatePropagation();this.controls.enabled=false;this.el.querySelector('canvas').setPointerCapture(e.pointerId);
   this.drag={...d,plane:new THREE.Plane(new THREE.Vector3(0,0,1),-d.z),moved:false};
  }
 }
 move(e){if(!this.drag)return;this.projectPointer(e);const p=new THREE.Vector3();if(this.ray.ray.intersectPlane(this.drag.plane,p)){this.drag.moved=true;this.onMove(this.drag.type,this.drag.index,[Math.round(p.x*10)/10,Math.round(p.y*10)/10,this.drag.z],false);}}
 up(){if(this.drag){const d=this.drag;this.drag=null;this.controls.enabled=true;if(d.moved)this.onMove(d.type,d.index,null,true);}}
 clear(group){group.traverse(o=>{o.geometry?.dispose();if(o.material){o.material.map?.dispose();o.material.dispose();}});group.clear();}
 label(text,color){const c=document.createElement('canvas');c.width=512;c.height=112;const ctx=c.getContext('2d');ctx.fillStyle='rgba(255,255,255,.93)';ctx.beginPath();ctx.roundRect(8,7,496,96,16);ctx.fill();ctx.fillStyle=color;ctx.font='600 40px Microsoft YaHei, sans-serif';ctx.textAlign='center';ctx.fillText(text,256,69);const texture=new THREE.CanvasTexture(c);const sprite=new THREE.Sprite(new THREE.SpriteMaterial({map:texture,depthTest:false}));sprite.center.set(.5,-.2);sprite.scale.set(72,16,1);sprite.renderOrder=20;return sprite;}
 update(config,selected){
  this.config=config;this.clear(this.devices);this.clear(this.links);this.items=[];
  for(const type of ['emitters','receivers'])config[type].forEach((d,index)=>{
   const pos=new THREE.Vector3(...d.position_m),rx=type==='receivers',col=rx?colors.receiver:colors[d.role],active=selected?.type===type&&selected?.index===index;
   const marker=new THREE.Mesh(rx?new THREE.ConeGeometry(active?5.5:4.5,12,4):new THREE.SphereGeometry(active?5.5:4.5,16,12),new THREE.MeshStandardMaterial({color:col,emissive:col,emissiveIntensity:.18,depthTest:false}));
   if(rx)marker.rotation.x=Math.PI/2;marker.position.copy(pos);marker.renderOrder=10;marker.userData={type,index,z:pos.z};this.devices.add(marker);this.items.push(marker);
   const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(pos.x,pos.y,.5),pos]),new THREE.LineDashedMaterial({color:col,dashSize:3,gapSize:2,transparent:true,opacity:.55}));line.computeLineDistances();this.devices.add(line);
   const ring=new THREE.Mesh(new THREE.RingGeometry(6,7.5,40),new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:.6,side:THREE.DoubleSide,depthTest:false}));ring.position.set(pos.x,pos.y,.4);this.devices.add(ring);
   if(this.showLabels){const lbl=this.label(`${d.id} · ${pos.z} m`,'#'+col.toString(16).padStart(6,'0'));lbl.position.copy(pos);this.devices.add(lbl);}
   if(rx){const [yaw,pitch,roll]=d.orientation_rad;const dir=new THREE.Vector3(1,0,0).applyEuler(new THREE.Euler(roll,pitch,yaw,'ZYX'));this.devices.add(new THREE.ArrowHelper(dir,pos,27,col,6,3));}
  });
  if(this.showLinks)for(const tx of config.emitters)for(const rx of config.receivers){const line=new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...tx.position_m),new THREE.Vector3(...rx.position_m)]),new THREE.LineDashedMaterial({color:0x91a5c4,dashSize:5,gapSize:4,transparent:true,opacity:.5}));line.computeLineDistances();this.links.add(line);}
 }
 setRadioMap(payload,metric){
  this.clearRadioMap();
  if(!payload||!payload.metrics||!payload.metrics[metric])return false;
  const g=payload.grid,m=payload.metrics[metric],nx=g.nx,ny=g.ny,cell=g.cell_size_m;
  if(!nx||!ny)return false;
  const vmin=m.vmin===null?-140:m.vmin,vmax=m.vmax===null?-40:m.vmax,span=Math.max(vmax-vmin,1e-6);
  const pixels=new Uint8Array(nx*ny*4);
  for(let i=0;i<nx*ny;i++){
   const v=m.data[i],p=i*4;
   if(v===null||v===undefined||!Number.isFinite(v)){pixels[p+3]=0;continue}
   const [r,gc,b]=rampColor((v-vmin)/span);
   pixels[p]=r;pixels[p+1]=gc;pixels[p+2]=b;pixels[p+3]=214;
  }
  const texture=new THREE.DataTexture(pixels,nx,ny,THREE.RGBAFormat);
  texture.flipY=false;texture.magFilter=THREE.LinearFilter;texture.minFilter=THREE.LinearFilter;
  texture.colorSpace=THREE.SRGBColorSpace;texture.needsUpdate=true;
  const mesh=new THREE.Mesh(new THREE.PlaneGeometry(nx*cell,ny*cell),
   new THREE.MeshBasicMaterial({map:texture,transparent:true,depthWrite:false,side:THREE.DoubleSide}));
  const [ox,oy]=g.origin_xy_m;
  mesh.position.set(ox+nx*cell/2,oy+ny*cell/2,g.height_m);
  mesh.renderOrder=6;
  mesh.userData={radioMap:true};
  this.radioMap=mesh;
  this.radioMapData={payload,metric,nx,ny,cell,origin:g.origin_xy_m,height:g.height_m};
  this.scene.add(mesh);
  this.applyOverlayStyle();
  return true;
 }
 clearRadioMap(){
  if(this.radioMap){
   this.scene.remove(this.radioMap);
   this.radioMap.geometry.dispose();
   this.radioMap.material.map?.dispose();
   this.radioMap.material.dispose();
   this.radioMap=null;this.radioMapData=null;
  }
  this.applyOverlayStyle();
 }
 toggleRadioMap(on){if(this.radioMap)this.radioMap.visible=on;this.applyOverlayStyle();}
 setRadioMapMode(mode){this.radioMapMode=mode==='ground'?'ground':'ghost';this.applyOverlayStyle();}
 applyOverlayStyle(){
  // 热力图是贴在测量高度上的一张水平面：建筑不透明时，低角度看过去会被楼挡掉。
  // 「透视建筑」把建筑改成半透明、不写深度、并排到热力图之后绘制，
  // 于是热力图完整可见、城市轮廓也还在；「贴地」恢复严格遮挡。
  const on=!!(this.radioMap&&this.radioMap.visible),ghost=on&&this.radioMapMode==='ghost';
  if(this.buildings){
   const b=this.buildings.material;
   b.transparent=ghost;b.opacity=ghost?0.25:1;b.depthWrite=!ghost;b.needsUpdate=true;
   this.buildings.renderOrder=ghost?8:0;
  }
  if(this.edges){this.edges.material.opacity=ghost?0.4:0.3;this.edges.renderOrder=ghost?9:0;}
 }
 toggleRadioMap(on){if(this.radioMap)this.radioMap.visible=on;}
 pickRadioMap(e){
  const d=this.radioMapData;
  if(!d||!this.radioMap||!this.radioMap.visible)return null;
  this.projectPointer(e);
  const hit=this.ray.intersectObject(this.radioMap,false)[0];
  if(!hit)return null;
  const col=Math.floor((hit.point.x-d.origin[0])/d.cell),row=Math.floor((hit.point.y-d.origin[1])/d.cell);
  if(col<0||row<0||col>=d.nx||row>=d.ny)return null;
  const i=row*d.nx+col,metric=d.payload.metrics[d.metric],txIndex=d.payload.association.data[i];
  return {x:hit.point.x,y:hit.point.y,row,col,value:metric.data[i],unit:metric.unit,label:metric.label,
          txId:(txIndex===null||txIndex===undefined)?null:d.payload.association.tx_ids[txIndex]};
 }
 toggleBuildings(on){if(this.buildings)this.buildings.visible=on;if(this.edges)this.edges.visible=on;}
}
